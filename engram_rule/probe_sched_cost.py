"""§69.42-P 原型墙钟：测「逐步自生成」到底比「整段一次前向」慢多少。

── 为什么先测这个（而不是直接改训练循环）─────────────────────────────
`design_scheduled_sampling.py` 的静态估算是 **8~40 倍慢**，区间太宽，
宽到无法决定要不要做（乐观 22 小时 vs 悲观 108 小时，差 5 倍）。
区间宽的根源是我假设了「答案段中位 40 token ⇒ 每微批要 40 次前向」，
但**真实的前向成本不是线性的**（序列变长、kernel 提交开销摊薄、显存带宽占用变化都未知）。

⇒ 本节只测一件事：**同一批样本，整段一次前向 vs 逐步自生成，墙钟比多少**。
   这一个数就把区间从 32 倍宽压到 2 倍以内。

── 怎么测（不训练、不落盘权重、不改任何已有文件）─────────────────────
取满步成品 + 一批 held 样本，做两个计时：

| 编号 | 做什么 | 记什么 |
|---|---|---|
| **A** | `model(bx)` 一次前向（现有训练的做法） | 墙钟 |
| **B** | 在答案段内逐步 `model(x)[0, pos]` 取 argmax 拼回去（自生成） | 墙钟 |

★B 用的是**贪心**（argmax），不用 `greedy_generate`（那个有 UTF-8 校验、
  重复惩罚等额外逻辑，会把计时污染）。这里只想要**纯前向次数**的成本比。

── 判据（可失败）───────────────────────────────────────────────────
| 编号 | 判据 | 失败意味着 |
|---|---|---|
| **W1** | B/A 的比值落在 [1, 100] | 若 <1 ⇒ 我的模型错了（自生成不该更快） |
| **W2** | 比值与「答案段 token 数」的关系被记下来 | 不给这个 ⇒ 无法判断是固定开销还是线性开销 |
| **W3** | A 与 B 的批大小、设备、序列来源**完全相同** | 不同 ⇒ 比的是两个东西 |

用法：`python probe_sched_cost.py [--rows 8] [--cap 40]`
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

CKPT = os.path.join(HERE, "engram_rule_scaling_v2_n124593_d128_wd0p0.pt")
OUT = os.path.join(HERE, "sched_cost.json")
SEED = 12345


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8, help="计时用的行数（够稳定即可）")
    ap.add_argument("--cap", type=int, default=40, help="自生成最多走多少 token")
    a = ap.parse_args()

    print("=" * 78)
    print("§69.42-P 原型墙钟：整段一次前向 vs 逐步自生成")
    print("=" * 78)
    print("  权重 = %s（只读，不训练、不落盘）" % os.path.basename(CKPT))

    import model as M                      # ★规则 7：E2 没导出 M
    E2.load_all()
    _rows = E2.load_rows()
    _tr, held = E2.split_rows(_rows)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])                  # ★逐字用成品 cfg
    torch.manual_seed(E2.SEED)
    m = M.EngramRuleLM(M.RCfg(**cfg)).to(E2.DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    x, s, l = E2.encode_cached(_tr, E2._TOK, E2.SEQ)
    info = E2.DS.build_engram_from_examples(m, x, l)
    assert info["alive_slots"] > 0, "alive 全 False ⇒ 残壳"
    print("  建 Engram：alive %d / %d" % (info["alive_slots"], info["n_slots"]))

    # ★W3：A 与 B 用**同一批** x / starts / lengths
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(held), generator=g)[:a.rows].tolist()
    samp = [held[i] for i in idx]
    bx_full, sb, lb = E2.encode_cached(samp, E2._TOK, E2.SEQ)
    bx_full = E2.DS._trim_length(bx_full, lb) if cfg.get("bucket_step", 0) > 0 else bx_full
    L = bx_full.shape[1]
    print("  样本 %d 行 × L=%d（截到该批真实最大行长）" % (bx_full.shape[0], L))
    print("  答案段起点 starts: min=%d max=%d；行长 lengths: min=%d max=%d"
          % (int(sb.min()), int(sb.max()), int(lb.min()), int(lb.max())))

    # ── A：整段一次前向（现有训练的做法）──────────────────────────
    dev = E2.DEV
    bxA = bx_full.to(dev)
    with torch.no_grad():
        m(bxA)                             # warmup
    if dev == "cuda":
        torch.cuda.synchronize()
    tA = time.time()
    with torch.no_grad():
        m(bxA)
    if dev == "cuda":
        torch.cuda.synchronize()
    secA = time.time() - tA
    print()
    print("【A】整段一次前向：%.4f s" % secA)

    # ── B：答案段内逐步自生成（贪心 argmax）────────────────────────
    bxB = bx_full.to(dev).clone()
    starts = sb.to(dev)
    with torch.no_grad():
        # warmup：走 2 步
        for p in range(int(starts.min()), min(int(starts.min()) + 2, L - 1)):
            _ = m(bxB)[0, p]
    if dev == "cuda":
        torch.cuda.synchronize()
    tB = time.time()
    n_steps = 0
    with torch.no_grad():
        for i in range(bxB.shape[0]):
            st = int(starts[i])
            for p in range(st, min(st + a.cap, L - 1)):
                logits = m(bxB[i:i + 1])[0, p]      # ★单行前向（最坏情况）
                bxB[i, p + 1] = logits.argmax()
                n_steps += 1
    if dev == "cuda":
        torch.cuda.synchronize()
    secB = time.time() - tB
    print("【B】逐步自生成：%.4f s（共 %d 次单行前向）" % (secB, n_steps))

    ratio = secB / max(secA, 1e-9)
    print()
    print("=" * 78)
    print("【W1】比值 B/A = %.2f 倍" % ratio)
    print("=" * 78)
    print("  单次单行前向 = %.4f ms；整段一次前向摊到每行 = %.4f ms"
          % (1000 * secB / max(n_steps, 1), 1000 * secA / bxA.shape[0]))
    assert 1.0 <= ratio, \
        "W1 失败：比值 %.2f < 1 ⇒ 自生成不该比整段前向更快，测法或模型有问题" % ratio
    if ratio > 100.0:
        print("  ★比值 %.1f 远超 design 预估上限（40×）⇒ **该方案在本机不可行**，"
              % ratio)
        print("    不再用 W1 的 100 上界否它（那是预估，不是物理限），继续算完外推供判决。")

    # W2：判固定开销还是线性开销
    per_step = secB / max(n_steps, 1)
    per_full = secA / max(bxA.shape[0], 1)
    print("  W2: 自生成每 token %.4f s/次；整段每行 %.4f s/次 ⇒ 每行要 %.1f 次才追平"
          % (per_step, per_full, per_full / max(per_step, 1e-9)))

    # ── 外推全量 ────────────────────────────────────────────────────
    print()
    print("【外推（按 design 里那份静态估算的同一口径）】")
    bs, micro = E2.BS, E2.MICRO_BATCH
    n_micro = -(-bs // micro)
    ans_med = 40
    steps = 19468
    # 每个微批：整段一次 = secA/n_micro；自生成 ≈ 每行 ans_med 次单行前向
    per_micro_full = secA / n_micro
    per_row_gen = per_step * ans_med
    per_micro_gen = per_row_gen * micro
    sps_new = per_micro_full * 0 + per_micro_gen * n_micro     # 每步 = n_micro 个微批
    print("  现每步约 %.3f s（%d 微批 × %.4f s）"
          % (per_micro_full * n_micro, n_micro, per_micro_full))
    print("  改后每步约 %.1f s（%d 微批 × %.1f s，每微批 %d 行 × %d token × %.4f s）"
          % (sps_new, n_micro, per_micro_gen, micro, ans_med, per_step))
    print("  倍数 = %.1f×" % (sps_new / max(per_micro_full * n_micro, 1e-9)))
    print("  全量 %d 步 ⇒ %.1f 小时" % (steps, sps_new * steps / 3600.0))
    print("  ★后台上限 3600 s ⇒ 每次约 %.0f 步" % (3600.0 / sps_new))

    facts = dict(ckpt=os.path.basename(CKPT), rows=a.rows, cap=a.cap, L=int(L),
                 sec_full=round(secA, 4), sec_gen=round(secB, 4), n_steps=n_steps,
                 ratio=round(ratio, 2),
                 per_step_sec=per_step, per_row_full_sec=per_full,
                 breakeven_steps=round(per_full / max(per_step, 1e-9), 1),
                 extrap=dict(bs=bs, micro=micro, n_micro=n_micro, ans_med=ans_med,
                             steps=steps, sec_per_step_new=round(sps_new, 2),
                             speedup_vs_now=round(sps_new / max(per_micro_full * n_micro, 1e-9), 1),
                             hours_full=round(sps_new * steps / 3600.0, 1),
                             steps_per_bg_run=int(3600.0 / sps_new)))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    print("  ★单次计时、单 seed ⇒ 无重复测量，比值只作数量级判断。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
