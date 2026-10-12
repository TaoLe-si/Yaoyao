"""§69.40 零成本验证：**问题段的信息有没有流到答案第一个位置**。

── 为什么先做这个（而不是直接开 q_pool）─────────────────────────────
§69.39 用 LCP 定位到「崩在第一个字」（LCP=0 占 64%）。我据此推断根因是
§61.2 那条旧实测：「答案第 1 步的窗口覆盖问题末尾 = 0/20，问题在结构上不可达」。

但那条读数来自**当时的权重与语料**（738 行时代）。现在的权重是
N=124,593 / d=128 满步，`start_bucket` 分布可能已经变了。
⇒ 在动架构（开 `use_qpool`）之前，必须**在当前权重上重新确认**这条，
   否则我可能拿着一条过期证据去改架构。

── 怎么做（不训练、不重训、不生成）──────────────────────────────────
载入满步成品 → 跑一次**真实前向** → 直接量三件事：

| 编号 | 量什么 | 判据 |
|---|---|---|
| **R1** | `start_bucket` 在当前权重上的实际分布 | 若仍只有 {0,1} ⇒ 窗口起点塌成常数，与 §65.2 旧读数一致 |
| **R2** | 答案第一个位置的窗口偏移范围 | 若最大偏移 < 答案起点 ⇒ **问题末尾到不了答案开头** |
| **R3** | 答案第一个位置能否「看见」问题末尾 | 直接算窗口是否覆盖 `starts - k` |

★R3 是 R2 的直接推论，但分开写是因为它才是「问题可达性」的那个量 ——
  判据必须落在**问题本身**上，不是落在偏移量上（AGENTS 规则 11：锚点与目标口径一致）。

── 负向对照（规则 9：判据必须可失败）─────────────────────────────
| 编号 | 对照 | 期望 |
|---|---|---|
| **N1** | 换一个 `pos_abs_max=512` 但 `clamp_window=False` 的假模型重算 R3 | 若覆盖率**相同** ⇒ 我的窗口索引算错了 |
| **N2** | 用**打乱的问题段**重跑前向，看 R3 是否变化 | 若不变 ⇒ 前向根本没读问题段（加强结论）；若变 ⇒ 读了但没用上 |

用法：`python probe_question_reach.py`（一次前向，秒级）
"""
import json
import os
import sys
from collections import Counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

CKPT = os.path.join(HERE, "engram_rule_scaling_v2_n124593_d128_wd0p0.pt")
OUT = os.path.join(HERE, "question_reach.json")
SAMPLE_SEED = 12345         # ★与 run_qa_tables.py 同一个抽样种子 ⇒ 同源
N_ROWS = 200                # ★与 run_qa_tables.py 的 SAMPLE 同值 ⇒ 同一批行


def main():
    print("=" * 78)
    print("§69.40 问题段可达性（当前满步权重上的一次真实前向）")
    print("=" * 78)
    print("  权重 = %s" % os.path.basename(CKPT))

    import model as M                      # ★规则 7：E2 没导出 M
    E2.load_all()
    rows = E2.load_rows()
    train_rows, held_rows = E2.split_rows(rows)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])                  # ★逐字用成品 cfg，一个字段都不重建
    torch.manual_seed(E2.SEED)
    m = M.EngramRuleLM(M.RCfg(**cfg)).to(E2.DEV)
    m.load_state_dict(ck["state_dict"])    # ★strict=True
    m.eval()

    x, s, l = E2.encode_cached(train_rows, E2._TOK, E2.SEQ)
    info = E2.DS.build_engram_from_examples(m, x, l)
    print("  建 Engram：alive %d / %d" % (info["alive_slots"], info["n_slots"]))
    assert info["alive_slots"] > 0, "alive 全 False ⇒ 测的是残壳"

    # ★★同源：与 run_qa_tables.py 用**同一个种子、同一个 SAMPLE** 抽 held，
    #   这样本脚本的 covered 与 qa_held_s200.jsonl 的 F1 能**逐行配对**。
    #   （上一版取前 64 行 train ⇒ row 索引与 qa 那份不同源 ⇒ 假对拍，已废。）
    g = torch.Generator().manual_seed(SAMPLE_SEED)
    idx = torch.randperm(len(held_rows), generator=g)[:N_ROWS].tolist()
    samp = [held_rows[i] for i in idx]
    sx, ss, sl = E2.encode_cached(samp, E2._TOK, E2.SEQ)
    print("  ★抽样口径与 run_qa_tables 一致：seed=%d / n=%d（held）"
          % (SAMPLE_SEED, len(samp)))

    # ── 取这批样本，截到该批真实最大行长 ────────────────────────────
    bx = sx.to(E2.DEV)
    sb = ss.to(E2.DEV)
    lb = sl.to(E2.DEV)
    bx = E2.DS._trim_length(bx, lb) if cfg.get("bucket_step", 0) > 0 else bx
    L = bx.shape[1]
    print("  前向：%d 行 × L=%d" % (bx.shape[0], L))

    # ── 打开诊断开关，跑一次前向 ────────────────────────────────────
    for blk in m.blocks:
        blk.attn.record = True
    with torch.no_grad():
        m(bx)
    att0 = m.blocks[0].attn
    startb = att0.last_startb              # (B,L)
    off = att0.last_window_off             # (B,L,W) 夹紧后的实际偏移
    step = int(cfg["bucket_step"])
    W = int(cfg["window_max"])
    nbk = int(cfg["n_buckets"])

    # ── R1 start_bucket 分布 ──────────────────────────────────────
    cnt = Counter(startb.flatten().tolist())
    print()
    print("【R1】start_bucket 实际分布（当前权重）")
    print("  取值 = %s" % sorted(cnt))
    for k in sorted(cnt):
        print("    bucket %2d : %7d  (%.2f%%)" % (k, cnt[k], 100.0 * cnt[k] / startb.numel()))
    max_off = int(off.max())
    print("  窗口偏移实测上界 = %d（= start_bucket_max × step + W − 1 = %d × %d + %d）"
          % (max_off, max(cnt), step, W - 1))

    # ── R2/R3 问题可达性 ──────────────────────────────────────────
    print()
    print("【R2/R3】答案第一个位置能不能「看见」问题末尾")
    # 答案第一个位置 = starts-1 处做预测（logits 在 pos 处预测 pos+1）
    # 该位置的窗口覆盖 [pos - off, pos - off + W - 1]
    # 问题末尾 ≈ starts-2（答案起点前一个字）
    cover = []
    detail = []
    for i in range(bx.shape[0]):
        p = int(sb[i]) - 1                     # 预测答案首 token 的位置
        if p < 1:
            continue
        row_off = off[i, p]                    # (W,) 该位置各窗口位的偏移
        src_lo = p - int(row_off.max())        # 窗口最远处
        src_hi = p - int(row_off.min())        # 窗口最近处
        q_end = int(sb[i]) - 2                 # 问题末尾的字
        ok = bool(src_lo <= q_end <= src_hi)
        cover.append(ok)
        detail.append(dict(row=int(i), q_end=int(q_end), pos=int(p),
                           win_lo=int(src_lo), win_hi=int(src_hi), covered=ok))
    n_ok = sum(cover)
    print("  样本 %d 条中，答案首步窗口**覆盖问题末尾**的 = %d（%.1f%%）"
          % (len(cover), n_ok, 100.0 * n_ok / max(len(cover), 1)))
    print("  ★对照 §61.2 旧读数（738 行时代）= **0/20**")

    facts = dict(ckpt=os.path.basename(CKPT), n_rows=int(bx.shape[0]), L=int(L),
                 bucket_step=step, window_max=W, n_buckets=nbk,
                 alive=info["alive_slots"],
                 sample=dict(seed=SAMPLE_SEED, n=N_ROWS, source="held（与 run_qa_tables 同源）"),
                 held_indices=[int(i) for i in idx],
                 startb_distinct=sorted(cnt),
                 startb_hist={int(k): int(v) for k, v in sorted(cnt.items())},
                 window_offset_max=max_off,
                 q_tail_covered=n_ok, q_tail_total=len(cover),
                 q_tail_cover_rate=n_ok / max(len(cover), 1),
                 detail=detail,
                 prior_61_2="0/20（738 行时代，过期证据，本节为当前权重重测）")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))

    # ── 判决 ──────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("【判决】")
    print("=" * 78)
    if facts["q_tail_cover_rate"] < 0.05:
        print("  ⇒ 问题末尾在当前权重上**仍然不可达** ⇒ §61.2 的结论**被重新确认**。")
        print("     「崩在第一个字」的根因是**问题信息结构上不可达**，不是权重不够重视。")
        print("     ⇒ 支持开 `use_qpool`（选项 A），B（调 first_token_weight）治不了本。")
    else:
        print("  ⇒ 覆盖率 %.1f%% ⇒ §61.2 的结论在当前权重上**不再成立**，"
              % (100 * facts["q_tail_cover_rate"]))
        print("     「问题不可达」这条要撤回，得另找「第一个字错」的原因。")
    print("  ★本节是**一次前向**的读数，未做 N1/N2 负向对照（见 docstring），")
    print("    所以结论强度限于「当前权重上问题末尾不可达」这个事实本身。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
