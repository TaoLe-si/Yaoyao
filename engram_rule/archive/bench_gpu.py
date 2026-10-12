"""微基准: 分辨"GPU 被抢占" 与 "我的代码慢"。

  1) 参照负载: 固定 matmul 循环, 报实测 TFLOPS。RTX 4070 Laptop FP32 峰值约 30 TFLOPS。
     若实测远低于峰值的 ~30%, 说明 GPU 在被别人占。
  2) 模型前向: 小 batch (bs=8) 单次前向耗时, 换算成每样本 ms 与 token/s。
     这是 AGENTS.md 规则 20 要求的 profile 读数(CUDA 实测, 不猜)。
"""
import time

import torch

from model import RCfg, EngramRuleLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PEAK_TFLOPS = 30.0     # RTX 4070 Laptop, FP32 非 tensor-core 峰值(约值, 用来判断是否被抢)


def bench_matmul(n=2048, iters=30):
    a = torch.randn(n, n, device=DEV)
    b = torch.randn(n, n, device=DEV)
    for _ in range(5):
        a @ b
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        a @ b
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return 2 * n ** 3 * iters / dt / 1e12, dt / iters * 1e3


def bench_model(ck, bs=8, seq=512, reps=5):
    cfg = RCfg(**ck["cfg"])
    m = EngramRuleLM(cfg).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    x = torch.randint(0, cfg.vocab, (bs, seq), device=DEV)
    with torch.no_grad():
        for _ in range(2):
            m(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            m(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
    per = (time.time() - t0) / reps
    return per * 1e3, per / bs * 1e3, bs * seq / per


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ck = torch.load(os.path.join(here, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    print("DEV =", DEV, "| torch", torch.__version__)
    if DEV != "cuda":
        print("无 GPU, 跳过参照对比。")
        return

    tf, ms = bench_matmul()
    frac = tf / PEAK_TFLOPS
    print("\n[参照] matmul 2048^3 x30 : %.2f TFLOPS (单次 %.2f ms)" % (tf, ms))
    print("       占 FP32 峰值(~%.0f TFLOPS)的 %.0f%%  => %s"
          % (PEAK_TFLOPS, 100 * frac,
             "GPU 可用" if frac > 0.3 else "★GPU 被其它负载抢占, 加速比不可信"))

    if DEV == "cuda":
        print("       显存: %.0f MiB / %.0f MiB" % (
            torch.cuda.memory_allocated() / 2**20, torch.cuda.mem_get_info()[1] / 2**20))

    for bs in (8, 32):
        tot, per_s, tps = bench_model(ck, bs=bs)
        print("\n[模型] bs=%-3d L=512 : 每批 %.1f ms | 每样本 %.2f ms | %.0f token/s"
              % (bs, tot, per_s, tps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())