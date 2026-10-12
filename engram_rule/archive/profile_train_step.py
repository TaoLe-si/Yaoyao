"""训练步 profile: §51 只测了生成, 训练"没收益"这个结论要重新用**出货训练路径**测。

为什么重测: §51.8b 的基准是自造的循环(bs=256/seq=256/全词表 CE), **不是**出货路径 ——
出货路径是 `train_dialogue` + `answer_loss_sums` + micro_batch 切分, SEQ=512。
两者的中间张量组成不同(出货多一份 answer-mask 与 first-token 加权), 结论不能互借。

**预登记(可证伪)**, 动手前写死(规则 1/7):

T1  训练步是**显存带宽受限**, 不是 launch-bound。
    机制: SwiGLU 窗口链物化 (B,L,W,H,Dg) 与 Engram 读出 (B,L,R,d), micro=64/L=512 下
    单个就有 64·512·16·4·16 = 33.5M 元素 = 134 MB, 一层里来回数个。
    **否证**: profiler 的 kernel CUDA 忙时占 wall < 40% ⇒ T1 被否(那还是 host 受限)。

T2  单步时间随 micro_batch **增大而下降**(到显存上限前), 因为算子提交次数按步数摊薄。
    **否证**: micro_batch 32→64 每步时间不变或变差 ⇒ T2 被否。

T3  `(B,L,W,H,Dg)` 类中间张量的元素数 **≥** `(B,L,R,d)` 类。
    **否证**: Engram 张量更大 ⇒ T3 被否(那说明该先优化 Engram 读出)。

T4  head 的 `(B,L,V)` + 它的 loss 是显著成本(V=1024 与 W·H·Dg=1024 同量级)。
    **否证**: 去掉 head+loss 的时间占比 < 10% ⇒ T4 被否。

★ 本脚本**只测不改**。所有读数走 `train_dialogue` 这条出货路径。
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
SEQ = 512            # 出货口径: exp_capacity_scaling.SEQ
BS = 256             # 规则 18
STEPS = 6            # 只测速度, 不测收敛
D_LIST = (64, 128)
MICROS = (16, 32, 64)


def make_cfg(ck, d):
    from model import RCfg
    c = dict(ck["cfg"])
    c["d"] = d
    c["n_heads"] = d // 16          # §48: head_dim 恒 16
    return RCfg(**c)


def make_data(n, seq, vocab, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(3, vocab, (n, seq), generator=g)
    starts = torch.full((n,), seq // 2, dtype=torch.long)
    lengths = torch.full((n,), seq, dtype=torch.long)
    return x, starts, lengths


def time_steps(ck, d, micro, data, steps=STEPS):
    """跑 `train_dialogue` 并计时。返回 (每步秒, 峰值 MiB)。"""
    from model import EngramRuleLM
    from dialogue_small import train_dialogue
    x, starts, lengths = data
    torch.manual_seed(7)
    m = EngramRuleLM(make_cfg(ck, d)).to(DEV)
    m.engram.alive[:] = True
    train_dialogue(m, x, starts, lengths, 1, BS, 0.003, DEV, seed=7,   # 预热
                   first_token_weight=1.0, micro_batch=micro)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_dialogue(m, x, starts, lengths, steps, BS, 0.003, DEV, seed=7,
                   first_token_weight=1.0, micro_batch=micro)
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if DEV == "cuda" else 0.0
    del m
    return dt, peak


def profile_step(ck, d, micro, data, reps=3):
    """单步的 profiler 读数: kernel CUDA 忙时 + 最贵的几个算子。

    ★T1 第一版读出 194% / 754%(>100%, 物理不可能) —— 是**我的求和写错了**:
      profiler 的 key_averages() 里 **aten 算子层与它底下的 CUDA kernel 层各记一次
      device time**, 全加就重复计了一遍。修法: 只累加非 `aten::` 开头的条目(真 kernel)。
      这就是规则 5 的第四次: 失败的是检查本身, 不是被测对象。
    """
    from torch.profiler import profile, ProfilerActivity
    from model import EngramRuleLM
    from dialogue_small import train_dialogue
    x, starts, lengths = data
    torch.manual_seed(7)
    m = EngramRuleLM(make_cfg(ck, d)).to(DEV)
    m.engram.alive[:] = True
    train_dialogue(m, x, starts, lengths, 1, BS, 0.003, DEV, seed=7,
                   first_token_weight=1.0, micro_batch=micro)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEV == "cuda" else [])
    with profile(activities=acts) as prof:
        train_dialogue(m, x, starts, lengths, 1, BS, 0.003, DEV, seed=7,
                       first_token_weight=1.0, micro_batch=micro)
        if DEV == "cuda":
            torch.cuda.synchronize()
    ka = prof.key_averages()
    kernel_us = sum(e.self_device_time_total for e in ka
                    if not e.key.startswith("aten::") and not e.key.startswith("cuda"))
    aten_us = sum(e.self_device_time_total for e in ka if e.key.startswith("aten::"))
    launches = sum(e.count for e in ka if e.key == "cudaLaunchKernel")
    top = sorted(((e.self_device_time_total, e.count, e.key) for e in ka
                  if e.self_device_time_total > 0), reverse=True)[:10]
    del m
    return dict(cuda_us=kernel_us, aten_us=aten_us, launches=launches, top=top)


def tensor_budget(ck, d, micro, seq):
    """T3/T4 的解析口径: 数一数各类中间张量的元素数。"""
    c = make_cfg(ck, d)
    B = micro
    W, H, Dg = c.window_max, c.n_heads, c.gate_rank
    R, dd, V = c.n_cand, d, c.vocab
    swiglu = B * seq * W * H * Dg
    engram = B * seq * R * dd
    head = B * seq * V
    return dict(window=swiglu, engram=engram, head=head,
                window_mb=swiglu * 4 / 2 ** 20, engram_mb=engram * 4 / 2 ** 20,
                head_mb=head * 4 / 2 ** 20)


def main():
    from bench_gpu import bench_matmul, PEAK_TFLOPS
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    print("DEV=%s torch=%s | 出货口径 SEQ=%d BS=%d steps=%d | vocab=%d window_max=%d "
          "bucket_step=%d n_buckets=%d"
          % (DEV, torch.__version__, SEQ, BS, STEPS, ck["cfg"]["vocab"],
             ck["cfg"]["window_max"], ck["cfg"]["bucket_step"], ck["cfg"]["n_buckets"]))
    if DEV != "cuda":
        print("无 GPU, 跳过 profiler 读数")
        return 1
    tf, _ = bench_matmul()
    gate = tf / PEAK_TFLOPS
    print("[门禁] 参照 matmul %.2f TFLOPS = 峰值的 %.0f%% => %s"
          % (tf, 100 * gate, "可用" if gate > 0.3 else "★受抢占影响: 只引用比值"))

    data = make_data(BS, SEQ, ck["cfg"]["vocab"])
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="64,128", help="逗号分隔的宽度列表")
    a = ap.parse_args()
    for d in [int(x) for x in a.d.split(",") if x.strip()]:
        print("\n===== d=%d =====" % d)
        tb = tensor_budget(ck, d, 64, SEQ)
        print("[T3/T4] micro=64 的单批张量: 窗口 (B,L,W,H,Dg)=%.1fM 元素/%.0f MB | "
              "Engram (B,L,R,d)=%.1fM/%.0f MB | head (B,L,V)=%.1fM/%.0f MB"
              % (tb["window"] / 1e6, tb["window_mb"], tb["engram"] / 1e6,
                 tb["engram_mb"], tb["head"] / 1e6, tb["head_mb"]))
        print("        窗口 %s Engram 的大 ⇒ T3 %s"
              % (">=" if tb["window"] >= tb["engram"] else "<",
                 "成立" if tb["window"] >= tb["engram"] else "被否"))
        print("        head 占窗口的 %.2f ⇒ T4 %s"
              % (tb["head"] / max(tb["window"], 1), "成立" if tb["head"] / max(tb["window"], 1) > 0.1
                 else "被否(<10%)"))
        for micro in MICROS:
            dt, peak = time_steps(ck, d, micro, data)
            print("  micro_batch=%-3d 每步 %6.1f ms | 峰值 %6.0f MiB" % (micro, dt * 1e3, peak))
        base = time_steps(ck, d, 64, data)[0]
        small = time_steps(ck, d, 32, data)[0]
        print("  [T2] micro 32→64 每步 %.1f→%.1f ms ⇒ %s"
              % (small * 1e3, base * 1e3,
                 "成立(变大更快)" if base < small else "被否(变大不更快)"))
        pf = profile_step(ck, d, 64, data)
        wall_us = base * 1e6
        print("  [T1] 单步 **kernel** CUDA 忙时 %.0f µs / wall %.0f µs = %.0f%% ⇒ %s"
              % (pf["cuda_us"], wall_us, 100 * pf["cuda_us"] / max(wall_us, 1e-9),
                 "带宽受限(T1 成立)" if pf["cuda_us"] / max(wall_us, 1e-9) > 0.4
                 else "★仍 host 受限(T1 被否)"))
        print("       (aten 算子层另计 %.0f µs —— 只用于对照, 不加进忙时, 否则会重复计)"
              % pf["aten_us"])
        print("  单步 cudaLaunchKernel %d 次" % pf["launches"])
        for us, cnt, key in pf["top"]:
            print("      %8.0f µs  ×%-4d %s" % (us, cnt, key))
    print("\n门禁 gate_frac=%.3f (引用速度比值前必须 >=0.3)" % gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
