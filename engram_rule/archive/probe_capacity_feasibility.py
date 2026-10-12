"""实验 A 的可行性探针: 各档 d 的参数量、跑得动的 micro_batch、训练耗时与显存峰值。

**不训练、不产出结论** —— 只回答"9 次实验跑不跑得动、要多久"。
口径锚点与 exp_capacity_scaling.py 完全一致(共用同一个 loo_and_subset / score_rows)。
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_PROBE = 6


def main():
    import exp_capacity_scaling as exp
    from dialogue_small import train_dialogue

    ck, tok, tr_rows, ev_rows, tr_pack, ev_pack = exp.load_data()
    tr_x, tr_s, tr_l = tr_pack
    tr_loo, tr_keep, _, _ = exp.loo_and_subset(tok, tr_rows, exp.SEQ)
    ev_loo, ev_keep, _, _ = exp.loo_and_subset(tok, ev_rows, exp.SEQ)
    print("train keep=%d (LOO=0:%d, LOO>=4:%d) | eval keep=%d"
          % (len(tr_keep), int((tr_loo == 0).sum()), int((tr_loo >= 4).sum()),
             len(ev_keep)), flush=True)

    if DEV != "cuda":
        print("DEV=cuda 不可用 —— 实验 A 无法在本机进行。")
        return 1

    total = torch.cuda.get_device_properties(0).total_memory / (1 << 20)
    print("\n=== R5 参数量 / Engram 占比 ===", flush=True)
    print("  %-5s %-7s %12s %12s %8s %10s" % ("d", "n_heads", "总参数", "Engram参数",
                                             "Engram%", "attn+ffn"), flush=True)
    for d in exp.D_MAIN:
        m = exp.make_model(ck, d, 11, tr_x, tr_l)
        tot = sum(p.numel() for p in m.parameters())
        en = sum(p.numel() for n_, p in m.named_parameters() if n_.startswith("engram"))
        print("  %-5d %-7d %12d %12d %7.1f%% %10d"
              % (d, d // exp.HEAD_DIM, tot, en, 100.0 * en / tot, tot - en), flush=True)
        del m
        torch.cuda.empty_cache()

    print("\n=== 训练耗时 / 显存 (每档跑 %d 步计时, 外推 200 步) ===" % STEPS_PROBE, flush=True)
    grand = 0.0
    for d in exp.D_MAIN:
        mb, probe_peak, probe_sec = exp.find_micro_batch(ck, d, 11, tr_pack)
        m = exp.make_model(ck, d, 11, tr_x, tr_l)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        train_dialogue(m, tr_x, tr_s, tr_l, STEPS_PROBE, exp.BS, 0.003, DEV, seed=11,
                       first_token_weight=1.0, micro_batch=mb)
        dt = (time.time() - t0) / STEPS_PROBE
        peak = torch.cuda.max_memory_allocated() / (1 << 20)
        mins = dt * 200 / 60.0
        grand += mins * 3
        flag = "  ⚠ 贴墙" if peak / total > 0.85 else ""
        print("  d=%-4d micro_batch=%-3d %.3f s/step -> 200 步 %5.1f min | 峰值 %6.0f MiB "
              "/ %.0f MiB (%4.1f%%)%s"
              % (d, mb, dt, mins, peak, total, 100.0 * peak / total, flag), flush=True)
        del m
        torch.cuda.empty_cache()

    print("\n  9 次主实验(3d × 3 seed)粗估: %.0f min; 3 次 C0 预检: %.0f min"
          % (grand, grand / 3.0), flush=True)
    print("  ⚠ 本机 GPU 空闲时仍被桌面合成进程抢占约 4×(§40), 实际可能更慢。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())