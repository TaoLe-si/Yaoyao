# -*- coding: utf-8 -*-
"""§69.30 稀疏注意力通路的静态设计 + 落盘事实来源。

── 为什么先写设计再动手（规则 1）─────────────────────────────────────
`audit_window_geometry.py` 已给出两条可证伪读数：
· 当前窗口最远只回看 **495 token**（W1/W2），L=1M 时覆盖 **0.0016%**；
· 几何采样 + **32 槽** 才够 1M（r≈1.8）；同 16 槽换对数只有 **8.3×**，差 244×。
⇒ 这一步要加 **一条不受 Engram 规则门控** 的稀疏通路（否则 `o = o*keep`
   一乘，该通路在无规则命中处仍是 0 —— §69.28.3 已证）。

── 可证伪预测（先写，再动手；失败即设计错）─────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **P1** | 关掉稀疏通路（`sparse_alphas` 置 0）⇒ 输出与旧版**逐位相同** | 不是纯加法 ⇒ 会改坏出货读数 |
| **P2** | 开通路后，位置 ≥ `reach_cont` 的行**有**非零梯度 | 参数没被接进计算图 |
| **P3** | 新参数在 `pos_abs_max` 边界外仍有梯度（不依赖位置表长度） | 长程通路仍被位置表卡住 |

── 与硬约束的核对（静态）───────────────────────────────────────────
· **现有 checkpoint 直接废掉**：新增 `nn.Linear(d, sparse_out*head_dim)` 与
  `nn.Embedding(2, sparse_out)` ⇒ state_dict 键变、且多了一个 module ⇒
  **不能** resume（判据口径见 `check_resume.py` 的 P3：恢复键/文件名必须覆盖全部自变量）。
  ⇒ 新通路要**单独开配置**跑，不掺进 scaling 网格；网格读数保持 `4faabefdcd58` 那套口径。
· **显存**：`(B,L,W,H,Dg)` 中间张量（规则 26 的 SwiGLU 教训）大小 ∝ `B·W`。
  出货 bs=256 / L=512 ⇒ `B·L = 131,072`。当前 W=16 已实测显存打满过（§40）。
  ⇒ 稀疏支路**必须复用**已 gather 出来的 `kk`/`vv`（`_window_index` 只产 16 格），
     且新张量按 `head_dim`（16）而不是 `gate_rank`（16）控制，避免再叠一层 5 维。
· **成本**：新增算子数应为 O(1)（每个 attention 一次 `einsum` + 一次 `unsqueeze` 广播），
  不新增 kernel 提交数（§51 教训：同步与提交才是 host 侧瓶颈）。

── 口径：为什么数字从这里出，而不是画图脚本里手打 ───────────────────
本脚本把上面每个数落成 `sparse_design_facts.json`，`draw_paper_arch.py` **只读 JSON**。
理由（吃过亏）：图上一处手打常量会在架构变更后静默印错，且没有任何机制会报警。

用法：`python design_sparse_attention.py`（纯静态，不加载模型）
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 出货 cfg（probe_cfg_reach.py 已实测）
N_BUCKETS, BUCKET_STEP, WINDOW_MAX = 16, 32, 16
POS_ABS_MAX = 512
D, N_HEADS, GATE_RANK = 64, 4, 16
REACH_CONT = (N_BUCKETS - 1) * BUCKET_STEP + (WINDOW_MAX - 1)      # = 495
OUT = os.path.join(HERE, "sparse_design_facts.json")


def main():
    print("=" * 78)
    print("§69.30 稀疏注意力通路的静态设计（只算数，不改代码）")
    print("=" * 78)

    # ── 1. 新参数与开销 ──────────────────────────────────────────────
    print()
    print("【1】要新增什么、多少参数、多少显存")
    N_LOCAL, N_GEO = 8, 8
    R_GEO = 2.0
    local = list(range(1, N_LOCAL + 1))
    geo = [int(BUCKET_STEP * R_GEO ** k) for k in range(N_GEO)]
    offsets = local + geo
    W_s = len(offsets)
    far = max(offsets)
    print("  本地稠密偏移   : %s" % local)
    print("  几何外推偏移   : %s  (step=%d × r=%.1f^k)" % (geo, BUCKET_STEP, R_GEO))
    print("  最远回看       : %d token" % far)

    p_sparse = W_s * D
    n_modules = 1
    print("  新增参数       : nn.Linear(%d, %d) 权重 %d 个（无 bias）" % (D, W_s * D, p_sparse))
    print("  新增 module 数 : %d（⇒ state_dict 键变 ⇒ 旧 checkpoint 不可 resume，已登记）"
          % n_modules)

    B, L = 256, 512
    mb_new = B * L * W_s * N_HEADS * GATE_RANK * 4 / 2**20
    mb_cont = B * L * WINDOW_MAX * N_HEADS * GATE_RANK * 4 / 2**20
    print("  中间张量       : (B=%d, L=%d, W_s=%d, H=%d, Dg=%d) = %.1f MB"
          % (B, L, W_s, N_HEADS, GATE_RANK, mb_new))
    print("  ★与现有连续窗口的 (B,L,W=%d,H,Dg)=%.1f MB 同量级 ⇒ 叠加后约 %.0f MB"
          % (WINDOW_MAX, mb_cont, mb_new + mb_cont))

    # ── 2. 覆盖率预测（可证伪）───────────────────────────────────────
    print()
    print("【2】P-覆盖率：加稀疏通路后，L=1M 的覆盖比例")
    cov_cont = 100.0 * WINDOW_MAX / 1_000_000
    cov_new = 100.0 * (WINDOW_MAX + W_s) / 1_000_000
    print("  仅连续窗口     : %.6f%%（%d / 1e6）" % (cov_cont, WINDOW_MAX))
    print("  连续 + %2d 稀疏 : %.6f%%（%d / 1e6）" % (W_s, cov_new, WINDOW_MAX + W_s))
    print("  ⇒ 覆盖率只提升 %.2f× —— ★**这不是本设计的卖点**" % (cov_new / cov_cont))
    print("     卖点是**最远距离**：%d → %d token（%.0f×），跨过 seq=2048 与 32K 两道坎"
          % (REACH_CONT, far, far / float(REACH_CONT)))

    # ── 3. 落到 1M 还需要什么（静态）─────────────────────────────────
    print()
    print("【3】离 1M 还差多少（纯算术）")
    for L_target in (2048, 32768, 1_000_000):
        ratio = L_target / float(far)
        print("  L=%-9d : %s（%.1f×）"
              % (L_target, "够，余量 %.1f×" % (far / float(L_target)) if ratio <= 1
                 else "还差 %.1f×" % ratio, max(ratio, 1.0 / ratio)))

    facts = dict(
        design="稀疏通路：本地稠密 + 几何外推，不受 Engram 规则门控",
        reach_cont=REACH_CONT,
        far=far,
        offsets=offsets,
        n_local=N_LOCAL,
        n_geo=N_GEO,
        r_geo=R_GEO,
        w_sparse=W_s,
        new_params=p_sparse,
        new_modules=n_modules,
        mb_new=round(mb_new, 1),
        mb_cont=round(mb_cont, 1),
        mb_total=round(mb_new + mb_cont, 1),
        cov_cont_pct=cov_cont,
        cov_new_pct=cov_new,
        speedup_cov=round(cov_new / cov_cont, 2),
        speedup_far=round(far / float(REACH_CONT), 1),
        bucket_step=BUCKET_STEP,
        window_max=WINDOW_MAX,
        n_buckets=N_BUCKETS,
        pos_abs_max=POS_ABS_MAX,
        d=D,
        n_heads=N_HEADS,
        gate_rank=GATE_RANK,
        predictions=dict(
            P1="sparse_alphas 置 0 => 输出与旧版逐位相同",
            P2="位置 >= %d 的行有非零梯度" % REACH_CONT,
            P3="pos_abs_max(%d) 边界外仍有梯度" % POS_ABS_MAX,
        ),
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 78)
    print("【设计结论（待实现，先登记预测）】")
    print("=" * 78)
    print("  ① 新增一条**不受 Engram 门控**的稀疏通路：本地稠密 %d + 几何 %d 槽"
          % (N_LOCAL, N_GEO))
    print("  ② 复用已 gather 的 kk/vv，不新增 5 维大张量（规则 26）")
    print("  ③ 新参数做成 `sparse_alphas`（长度 %d 的可学习标量），"
          "置 0 时输出与旧版逐位相同（P1）" % W_s)
    print("  ④ 新通路要过三道：P1 逐位等价、P2 有梯度、P3 位置表外仍有梯度")
    print("  ⑤ 旧 checkpoint **不可 resume**（state_dict 键变）—— 已登记，不掺进网格")
    print()
    print("  ★还没有动一行代码。这一步是**设计记账**，实现要等 P1/P2/P3 写进门禁之后。")
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
