"""§69.4 追加测量 **M4**：`int(lb.max())` 在**真实训练步里**到底阻塞多久。

**为什么必须补 M4**：M1 用「64 个 512×512 matmul」当队列深度的代理，量到最深队列下
每微批停顿 1.18 ms。但那个代理只有 **0.64 ms 的 GPU 活**，而真实步里调用
`int(lb.max())` 之前，队列里排着的是**上一个微批的完整前向 + 反向（约 89 ms）**。

⇒ 拿 M1 的数字线性外推到「真实队列深度」是**错的**，而且是往小了错。
M4 不外推，直接在出货训练步的同一段代码上插计时，量 host 被阻塞的真实时长。

★两个必须注意的点：
  1. 这里量的是**host 阻塞时长**，不是 GPU 干活时长 —— 前者是可优化掉的，
     后者不是（§68.7.4 的教训：把「墙钟占比」当「可消除开销」）。
  2. 计时点用 `time.perf_counter()`，它自己约 100 ns，相对毫秒级停顿可忽略；
     且**不改动** `dialogue_small.py`（计时只发生在本脚本的复刻里）。

用法：`python probe_sync_insitu.py [d]`
"""
import os
import statistics
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BS, MB = 256, 64
N_ROWS = 124593
WARMUP, N_STEP = 3, 10


def run(model, x_dev, s_dev, l_dev, idx_seq, steps, instrument):
    """复刻出货训练步；`instrument=True` 时统计 `int(lb.max())` 的 host 阻塞时长。

    ★与 `train_steps` 逐行同构，差别只有：
      · 去掉 `sched`（它与计时无关，却会在步数设短时直接抛错）
      · `_trim_length` 展开成 `int(lb.max())` + 切片，以便插计时点
    """
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    model.train()
    bucket_step = getattr(model.c, "bucket_step", 0)
    sync_s = []
    per_step_sync = []
    t_all = time.perf_counter()
    for step in range(steps):
        idx = idx_seq[step % len(idx_seq)]
        norm = (l_dev[idx] - s_dev[idx]).sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        sl = torch.zeros((), dtype=torch.float32, device=DEV)
        step_sync = 0.0
        for i in idx.split(MB):
            bx = x_dev[i]
            sb = s_dev[i]
            lb = l_dev[i]
            if instrument:
                t0 = time.perf_counter()
                keep = int(lb.max())
                step_sync += time.perf_counter() - t0
            else:
                keep = int(lb.max()) if bucket_step > 0 else bx.shape[1]
            if bucket_step > 0:
                bx = bx[:, :max(keep, 2)]
            pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            import exp_scaling_grid_v2 as e
            tl, _a, _b, _c = e.DS.answer_loss_sums(logits, bx[:, 1:], valid, sb)
            (tl / norm).backward()
            sl += tl.detach()
        opt.step()
        per_step_sync.append(step_sync)
        sync_s.append(step_sync)
    if DEV == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t_all
    return float(sl / norm), wall, per_step_sync


def main():
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    import exp_scaling_grid_v2 as e
    import model as M
    e.load_all()
    rows = e.load_rows()
    tr, he = e.split_rows(rows)
    e.assert_no_leak(tr, he)
    x, s, l = e.encode_cached(tr[:N_ROWS], e._TOK, 512)
    ck = e._CK

    print("=" * 74)
    print("M4 原地测量：真实训练步里 `int(lb.max())` 的 host 阻塞时长  d=%d  DEV=%s"
          % (d, DEV))
    print("=" * 74)

    def mk():
        torch.manual_seed(11)
        m = M.EngramRuleLM(M.RCfg(**{**ck["cfg"], "d": d, "n_heads": d // 16})).to(DEV)
        e.DS.build_engram_from_examples(m, x, l)
        return m

    x_dev, s_dev, l_dev = x.to(DEV), s.to(DEV), l.to(DEV)
    g = torch.Generator().manual_seed(11)
    idx_seq = [torch.randint(0, x.shape[0], (BS,), generator=g)
               for _ in range(N_STEP + WARMUP)]

    # 基线（不插计时）：拿到干净的 s/步
    m = mk()
    run(m, x_dev, s_dev, l_dev, idx_seq, WARMUP, False)
    base_loss, base_wall, _ = run(m, x_dev, s_dev, l_dev, idx_seq, N_STEP, False)
    base_per = base_wall / N_STEP
    print("  基线（无插桩）      %.4f s/步   末 loss %.6f" % (base_per, base_loss))
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # 插桩：同一段代码，只加两个 perf_counter
    m = mk()
    ins_loss, ins_wall, per_step_sync = run(m, x_dev, s_dev, l_dev, idx_seq,
                                            WARMUP, True)
    ins_loss2, ins_wall2, per_step_sync = run(m, x_dev, s_dev, l_dev, idx_seq,
                                              N_STEP, True)
    ins_per = ins_wall2 / N_STEP
    sync_per_step = statistics.median(per_step_sync)
    print("  插桩（同代码）      %.4f s/步   末 loss %.6f" % (ins_per, ins_loss2))
    print("  每步同步阻塞（中位） %.4f ms/步  %s"
          % (sync_per_step * 1000,
             [round(v * 1000, 2) for v in per_step_sync]))
    print("  插桩扰动            %+.2f%%（两个 perf_counter 的成本，应可忽略）"
          % (100.0 * (ins_per / base_per - 1.0)))
    print("  数值一致性          末 loss max|Δ| = %.3e"
          % abs(base_loss - ins_loss2))

    ceiling = sync_per_step / ins_per
    print("\n【判决】候选 e（跨流掩盖 `int(lb.max())` 同步）的天花板")
    print("  可消除（若完全掩盖） %.4f ms/步" % (sync_per_step * 1000))
    print("  单步墙钟             %.4f ms/步" % (ins_per * 1000))
    print("  ⇒ 天花板             **%.2f%%**" % (100.0 * ceiling))
    if ceiling < 0.02:
        print("  ⇒ < 2%：**判「无收益」**。§69.2 的 P7 要求 ≥ 1.02 ⇒ 不并入出货路径。")
        print("     （P8 等价性也不必再验：候选在收益上就已经出局。）")
    else:
        print("  ⇒ ≥ 2%%：候选**未被否决**，仍需跑 P8 等价性 + P7 交替 A/B 才能下结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
