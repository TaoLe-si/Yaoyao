"""训练加速总账: `trim_padding` 开/关, **真实数据**, 同机同种子背靠背。

数据走 `exp_capacity_scaling.load_data()` 那条出货路径(820 行 / SEQ=512 / 真实行长)。
先跑 warmup, 再各跑 N 步; 逐步 loss 必须一致(否则速度读数无意义)。
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
BS = 256
MICRO = 64
STEPS = 20
WARMUP = 2
D_LIST = (64, 128)


def build(ck, d, seed=11):
    from model import RCfg, EngramRuleLM
    cfg = dict(ck["cfg"])
    cfg["d"] = d
    cfg["n_heads"] = d // 16
    torch.manual_seed(seed)
    m = EngramRuleLM(RCfg(**cfg)).to(DEV)
    m.engram.alive[:] = True
    return m


def run(ck, d, tr, trim, steps=STEPS):
    from dialogue_small import train_dialogue
    x, s, l = tr
    m = build(ck, d)
    train_dialogue(m, x, s, l, WARMUP, BS, 0.003, DEV, seed=11,
                   first_token_weight=1.0, micro_batch=MICRO, trim_padding=False)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_dialogue(m, x, s, l, steps, BS, 0.003, DEV, seed=11,
                   first_token_weight=1.0, micro_batch=MICRO, trim_padding=trim)
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if DEV == "cuda" else 0.0
    del m
    return dt, peak


def mean_trimmed_L(tr, reps=200):
    """截断后每步实际用的 L 的均值(按微批取 max)。"""
    x, s, l = tr
    g = torch.Generator().manual_seed(11)
    lt = l.tolist()
    n = len(lt)
    tot = 0.0
    for _ in range(reps):
        idx = torch.randint(0, n, (BS,), generator=g).tolist()
        for i in range(0, BS, MICRO):
            tot += max(max(lt[j] for j in idx[i:i + MICRO]), 2)
    return tot / (reps * (BS // MICRO))


def main():
    import exp_capacity_scaling as ecs
    from bench_gpu import bench_matmul, PEAK_TFLOPS
    ck, tok, tr_rows, ev_rows, tr, ev = ecs.load_data()
    print("真实训练数据: %d 行 × %d | 行长 中位=%d max=%d"
          % (tr[0].shape[0], tr[0].shape[1],
             sorted(tr[2].tolist())[len(tr[2]) // 2], int(tr[2].max())))
    ml = mean_trimmed_L(tr)
    print("截断后每微批平均 L = %.0f (原 %d, 即 %.0f%%) ⇒ 理论上限 %.2f×"
          % (ml, SEQ, 100 * ml / SEQ, SEQ / ml))
    gate = float("nan")
    if DEV == "cuda":
        tf, _ = bench_matmul()
        gate = tf / PEAK_TFLOPS
        print("[门禁] 参照 matmul %.2f TFLOPS = 峰值的 %.0f%% => %s"
              % (tf, 100 * gate, "可用" if gate > 0.3 else "★受抢占影响, 只引用比值"))
    for d in D_LIST:
        off, p_off = run(ck, d, tr, False)
        on, p_on = run(ck, d, tr, True)
        print("\nd=%d micro=%d  不截断 %.1f ms/步 (峰值 %.0f MiB) | 截断 %.1f ms/步 (峰值 %.0f MiB)"
              " | **%.2f×**" % (d, MICRO, off * 1e3, p_off, on * 1e3, p_on,
                                 off / max(on, 1e-9)))

    # ---- 评测侧: 同一机制, 但必须实测而不是外推 ----
    from dialogue_small import evaluate_dialogue
    for d in D_LIST:
        r = []
        for trim in (False, True):
            m = build(ck, d)
            evaluate_dialogue(m, ev[0], ev[1], ev[2], 32, DEV, trim_padding=False)   # 预热
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            evaluate_dialogue(m, ev[0], ev[1], ev[2], 32, DEV, trim_padding=trim)
            if DEV == "cuda":
                torch.cuda.synchronize()
            r.append(time.perf_counter() - t0)
            del m
        print("评测 d=%d: 不截断 %.2f s | 截断 %.2f s | **%.2f×**"
              % (d, r[0], r[1], r[0] / max(r[1], 1e-9)))
    print("\n门禁 gate_frac=%.3f" % gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
