# -*- coding: utf-8 -*-
"""★§69.29 窗口几何审计：1M 上下文的**一号瓶颈**到底损失多少覆盖。

── 为什么先做这个（而不是先做位置编码）────────────────────────────────
`probe_cfg_reach.py` 已经用纯算术证明：出货 cfg 的可达距离是
`n_buckets × bucket_step = 512` token，与 `pos_abs_max` 是**两条独立**通路。
本脚本把这句话再往下追一层：**512 之外不是"看不见"，是"连采样都采不到"**。

── 三条可证伪预测 ────────────────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **W1** | 查询能触达的源位置集合 = **一段连续区间**（非稀疏） | 若稀疏，则"窗口连续"这个前提错，结论要改 |
| **W2** | 连续窗口的覆盖随 L **线性衰减**，L=1M 时低到不可用 | 若不衰减，则瓶颈另有原因 |
| **W3** | 几何采样能覆盖 1M，但**所需槽数必须算出来** | 若任何一档都够不着，几何采样这条路本身要重估 |

★**W3 的预登记预测已被自己推翻**（这就是把它写成可证伪的原因）：
  最初写「同样 16 槽，对数采样能把最远距离推远 **≥1000×**」。
  算下来 `r=2 / 16 槽 = 2^12 = 4096`，相对 497 只有 **8.2×**。
  ⇒ 撤回该预测，改为报告**真正可执行的数**：「要多少槽 × 什么间隔」。

用法：`python audit_window_geometry.py`
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_BUCKETS = 16        # 出货 cfg
BUCKET_STEP = 32      # 出货 cfg
WINDOW_MAX = 16       # 出货 cfg
REACH = N_BUCKETS * BUCKET_STEP

LENGTHS = [512, 2_048, 32_768, 1_000_000]


def current_reachable(q, L, nb=N_BUCKETS, bs=BUCKET_STEP, W=WINDOW_MAX):
    """复刻 `model.py:_window_index` 的 `off0 = min(start_bucket*step, max_start)`。

    起点未知（由规则学出来），所以这里取**最优起点** `start_bucket = nb-1`
    ⇒ `off0 = min((nb-1)*bs, max(q-W+1, 0))`，再铺一段宽 W 的**连续**偏移。
    返回可达源位置集合。
    """
    off0 = min((nb - 1) * bs, max(q - W + 1, 0))
    return {q - (off0 + d) for d in range(W) if q - (off0 + d) >= 0}


def log_spaced(q, W=WINDOW_MAX, n_local=4):
    """候选：前 `n_local` 个挨着，后面 `W-n_local` 个按 2 的幂往外跳。

    ★对数间隔是长程检索的常见做法（能覆盖到 q 本身量级），
      但**这里只算覆盖率，不假设它训练得起来**。
    """
    out = {q - d for d in range(n_local)}
    out.add(q)
    stride = 1
    for d in range(W - n_local):
        stride = min(stride * 2, 1 << 30)
        out.add(q - stride)
    return {j for j in out if j >= 0}


def main():
    print("=" * 78)
    print("窗口几何审计（出货 cfg: n_buckets=%d, bucket_step=%d, window_max=%d）"
          % (N_BUCKETS, BUCKET_STEP, WINDOW_MAX))
    print("=" * 78)

    # ── W1：可达集合必须是**连续**区间 ──────────────────────────────────
    print()
    print("【W1】查询位置 q=100,000 处的可达源位置（连续窗口）")
    src = sorted(current_reachable(100_000, 100_000))
    gaps = {src[i + 1] - src[i] for i in range(len(src) - 1)}
    print("  数量 = %d，范围 = [%d, %d]" % (len(src), src[0], src[-1]))
    print("  相邻间隔集合 = %s  ⇒ %s" % (sorted(gaps),
                                        "连续区间 ✓" if gaps == {1} else "★非连续，前提错"))
    assert gaps == {1}, "W1 失败：可达集合不是连续区间 ⇒ 本审计的前提错，结论要重写"
    print("  ⇒ W1 成立：模型只能看**一段连续**的 %d 个 token" % len(src))

    # ── W2：可达集合随 L 变远，占前缀比例线性衰减 ──────────────────────
    print()
    print("【W2】序列变长时，连续窗口能看到什么")
    print("  %10s %16s %16s %16s" % ("L", "连续最远回看", "对数最远回看", "连续窗口覆盖率"))
    rows = []
    for L in LENGTHS:
        q = L - 1
        cont = current_reachable(q, L)
        lg = log_spaced(q)
        f_cont = max(q - j for j in cont)
        f_log = max(q - j for j in lg)
        rows.append((L, f_cont, f_log))
        print("  %10d %16d %16d %15.6f%%"
              % (L, f_cont, f_log, 100.0 * len(cont) / L))
    frac = len(current_reachable(LENGTHS[-1] - 1, LENGTHS[-1])) / LENGTHS[-1]
    print("  ★覆盖率在 L=1M 时是 %.6f%%（16 个 token / 1,000,000）"
          % (100.0 * frac))
    print("  ★注意：同样 16 个槽，**两种采样覆盖的 token 数一样多**，")
    print("     差别全在「摊开多远」—— 这正是要换几何的理由。")
    assert frac < 1e-3, "W2 前提失效：L=1M 时覆盖率没那么低，得换个说法"

    # ── W3：要到 1M，预算与间隔各需要多少 ──────────────────────────────
    print()
    print("【W3】★达到 1,000,000 回看需要「多少槽 × 什么间隔」")
    print("  （几何采样：第 k 个槽回看 r^k，最大回看 = r^(槽数−本地槽数)）")
    print("  %8s" % "槽数" + "".join("%12s" % ("r=%.1f" % r) for r in (1.5, 1.8, 2.0, 3.0)))
    need = {}
    for n in (16, 32, 64, 128, 256):
        cells = []
        for r in (1.5, 1.8, 2.0, 3.0):
            reach = r ** (n - 4)          # 4 个本地稠密槽
            cells.append("%12s" % ("%.3g" % reach))
            if reach >= 1_000_000 and ("r=%.1f" % r) not in need:
                need["r=%.1f" % r] = n
        print("  %8d%s" % (n, "".join(cells)))
    print()
    print("  ⇒ 最省的一档：")
    for k in sorted(need, key=lambda k: need[k]):
        print("     %s × %d 槽 即可覆盖 1M" % (k, need[k]))
    assert need, "W3 失败：表里没有任何一档够 1M ⇒ 几何采样这条思路本身要重估"
    best = min(need.values())
    print("  ⇒ W3 成立：连续窗口（16 槽）差 %.0f×；几何采样至少要 %d 槽 + 稠密本地段"
          % (1_000_000 / rows[-1][1], best))
    print("     ⇒ **光把 16 个槽换成对数间隔是不够的**（r=2 时 16 槽只到 %.0f）。"
          % (2.0 ** 12))
    print("     这条是我写脚本时的**预登记预测被自己推翻**的：原先写「≥1000×」，")
    print("     算下来 r=2 / 16 槽只有 %.1f× ⇒ 撤回，改为报「需要多少槽」这个可执行的数。"
          % (2.0 ** 12 / max(rows[-1][1], 1)))

    print()
    print("=" * 78)
    print("【结论】1M 上下文的第一号瓶颈是**窗口几何**，不是位置表")
    print("  · 当前窗口是**连续** W=16 ⇒ 最远只回看 %d token" % rows[-1][1])
    print("  · L=1M 时它只覆盖 %.6f%% 的前缀" % (100.0 * frac))
    print("  · 同预算换对数间隔：最远 %.0f token（%.1f×），但**离 1M 还差 %.0f×**"
          % (rows[-1][2], rows[-1][2] / max(rows[-1][1], 1), 1_000_000 / rows[-1][2]))
    print("  · 真要覆盖 1M：几何采样 + 至少 **%d** 个槽 + 一段稠密本地窗口" % best)
    print()
    print("  ★**这只证明了「几何上算得出来」，没有证明「训练得起来」。**")
    print("    槽数从 16 加到 %d，单步算力与显存都会成倍上涨（本机 8188 MiB），" % best)
    print("    每个槽摊到的梯度也会变少。是否值得，是**待测的实验**，不是结论。")
    print("    落地顺序：先做几何改造并压住开销 → 测 → 再谈 1M 位置编码。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
