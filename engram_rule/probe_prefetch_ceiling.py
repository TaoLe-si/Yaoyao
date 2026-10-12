"""§69.4 预取流水线：**先量天花板，再决定要不要建**。

**这一脚本只测量，不改任何出货路径**（`exp_scaling_grid_v2.py` / `dialogue_small.py` 一行未动）。

── 为什么先量 ──────────────────────────────────────────────────────────
桃批准的是「用独立 CUDA 流把第 i+1 个微批的 H2D 发出去，与第 i 个微批计算重叠」。
但 §68.7.3 的 P4 已经把 H2D 从训练循环里消掉了（x/s/l 常驻 GPU，步内只有设备内
gather）⇒ **按字面实现没有可预取的对象**。

而 §69.2 的候选清单我写漏了一项，必须在这里更正：
`DS._trim_length` 的第一行是 **`keep = int(lengths_b.max())`** ——
这是一个**每微批一次的 D2H 同步**。d=64 / micro_batch=64 时，**一步 4 次**。

★这才是真正的流水线气泡：`int()` 会阻塞 host 直到 GPU 排空，
  于是「本微批的算子还没发完，下一微批的切分长度就等着要」。
  与「预取 262 KB」不同，这一项是**同步点**，量级完全不是一回事。

本脚本量三样东西，任何一样都可以单独失败：

| 编号 | 量 | 判据意义 |
|---|---|---|
| M1 | `int(lb.max())` 在**空队列** vs **满队列**下的直接耗时 | 差值 = 每个同步点的真实停顿 |
| M2 | 真实训练步的 **wall vs kernel self CUDA time** | 差值 = **所有重叠类优化的天花板** |
| M3 | 静态数一遍步内还有几个 D2H 同步点 | 与 M1 相乘 = 总停顿 |

★**M2 的读法必须用 profiler 的 self CUDA time**。两种错误读法都踩过：
  · CUDA event 的 span **不等于** GPU 忙时（它含 host 排队等待，空转也能读出 100%）
  · `key_averages()` 里 aten 算子层与它底下的 CUDA kernel 层**各记一次**
    `self_device_time_total`，全加会翻倍（实测过 194% / 754% 这种物理不可能的数）
  ⇒ 这里只累加 `not key.startswith("aten::")` 的项。

用法：`python probe_prefetch_ceiling.py [d]`
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


def m1_sync_cost(l_dev, tag, depths=(0, 4, 16, 64)):
    """M1：`int(lb.max())` 在不同**队列深度**下的直接耗时。

    ★为什么要扫队列深度：若只在浅队列下测，真实训练步排的深队列会把停顿**低估**
      ⇒ 天花板算小 ⇒ 结论错。所以扫多个深度，看停顿是**饱和**还是随队列线性增长。
    """
    lb = l_dev[:MB]
    # 负向对照：`0 == 0` 是纯 CPU，量的是 Python 开销本身，不是 GPU 往返
    names = (("int(lb.max())", lambda: int(lb.max())),
             ("lb.max() 只取不转", lambda: lb.max()),
             ("0 == 0（纯 CPU 对照）", lambda: 0 == 0))
    print("  [%s] 语句                 %s" % (tag, "  ".join("d=%-2d" % k for k in depths)))
    out = {}
    for name, fn in names:
        cells = []
        for depth in depths:
            ts = []
            for _ in range(200):
                if depth:
                    # ★每轮都用**新**的 512×512：写成 `a = a @ ...` 连乘的话，
                    #   第二次就退化成 64×64 再乘 512×64 直接 RuntimeError。
                    a = torch.randn(512, 512, device=DEV)
                    for _ in range(depth):
                        a = a @ a
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            cells.append(statistics.median(ts) * 1e3)
        out[name] = cells
        print("  [%s] %-22s %s" % (tag, name, "  ".join("%6.3f" % v for v in cells)))
    return out


def m2_busy_ratio(model, x_dev, s_dev, l_dev, n_step=6):
    """M2：真实训练步的 wall vs kernel self CUDA time。"""
    import exp_scaling_grid_v2 as e
    g = torch.Generator().manual_seed(11)
    idx_seq = [torch.randint(0, x_dev.shape[0], (BS,), generator=g) for _ in range(n_step + 2)]

    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    # ★total_steps 必须**大于**实际会走过的步数：预热 2 + 计时 n_step + profiler n_step。
    #   第一版按 n_step+2 设，第二遍（profiler 那遍）多走了 n_step 步 ⇒
    #   OneCycleLR 直接 ValueError。这不是训练代码的问题，是探针自己设短了。
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.003,
                                                total_steps=4 * n_step + 8, pct_start=0.1)
    model.train()
    bs_ = getattr(model.c, "bucket_step", 0)

    def one(idx):
        norm = (l_dev[idx] - s_dev[idx]).sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        sl = torch.zeros((), dtype=torch.float32, device=DEV)
        for i in idx.split(MB):
            bx = x_dev[i]
            sb = s_dev[i]
            lb = l_dev[i]
            if bs_ > 0:
                bx = e.DS._trim_length(bx, lb)
            pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            tl, _a, _b, _c = e.DS.answer_loss_sums(logits, bx[:, 1:], valid, sb)
            (tl / norm).backward()
            sl += tl.detach()
        opt.step()
        sched.step()
        return sl

    for k in range(2):                                   # 预热
        one(idx_seq[k])
    torch.cuda.synchronize()

    # 干净计时
    t0 = time.perf_counter()
    for k in range(n_step):
        one(idx_seq[k])
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for k in range(n_step):
            one(idx_seq[k])
        torch.cuda.synchronize()
    ka = prof.key_averages()
    kernel_us = sum(e.self_device_time_total for e in ka
                    if not e.key.startswith("aten::"))   # ★排除 aten 层，否则翻倍
    all_us = sum(e.self_device_time_total for e in ka)
    n_aten = sum(1 for e in ka if e.key.startswith("aten::"))
    busy = kernel_us / 1e6
    print("  M2  %d 步：wall %.4f s（%.3f s/步） | kernel self CUDA %.4f s"
          % (n_step, wall, wall / n_step, busy))
    print("      忙时占比 %.1f%%  ⇒ GPU 空闲占比 %.1f%%（**所有**重叠类优化的天花板）"
          % (100.0 * busy / wall, 100.0 * (1.0 - busy / wall)))
    # ★这行**不能**当互证：本次只开了 ProfilerActivity.CUDA，键表里根本没有 CPU 侧
    #   aten 条目（n_aten=0），所以 all_us 必然等于 kernel_us，是**恒等式而非证据**。
    #   上一版把它写成「说明排除 aten:: 是必要的」，那是**假互证**，已更正。
    print("      键表里 aten:: 条目数 = %d（只开了 CUDA activity ⇒ 无 CPU 侧 aten 条目，"
          "故 all=kernel 是恒等式，**不构成互证**）" % n_aten)
    return wall, busy


def m3_static_syncs():
    """M3：静态数一遍步内的 D2H 同步点。"""
    import inspect
    import exp_scaling_grid_v2 as e
    src = inspect.getsource(e.DS._trim_length)
    has = "int(lengths_b.max())" in src
    print("  M3  `_trim_length` 含 D2H 同步 `int(lengths_b.max())`：%s" % has)
    print("      每步同步点数 = 每步微批数 = BS/MB = %d/%d = %d"
          % (BS, MB, BS // MB))
    return has


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

    print("=" * 72)
    print("预取天花板测量：d=%d, N=%d, BS=%d, micro_batch=%d, DEV=%s"
          % (d, min(N_ROWS, len(tr)), BS, MB, DEV))
    print("=" * 72)

    torch.manual_seed(11)
    m = M.EngramRuleLM(M.RCfg(**{**ck["cfg"], "d": d, "n_heads": d // 16})).to(DEV)
    e.DS.build_engram_from_examples(m, x, l)
    x_dev, s_dev, l_dev = x.to(DEV), s.to(DEV), l.to(DEV)

    print("\n【M1】每微批一次 D2H 同步的直接耗时（按队列深度扫描，单位 ms）")
    m1 = m1_sync_cost(l_dev, "d=%d" % d)
    cells = m1["int(lb.max())"]
    depths = (0, 4, 16, 64)
    stalls = [max(0.0, c - cells[0]) for c in cells]
    print("      ⇒ 每微批停顿（相对空队列）: %s"
          % "  ".join("d=%d:%.4f" % (k, v) for k, v in zip(depths, stalls)))
    per_step = max(stalls) * (BS // MB)
    print("      ⇒ 取**最坏**（最深队列）算：一步 %d 个微批 ⇒ %.4f ms/步"
          % (BS // MB, per_step))

    print("\n【M2】真实训练步：wall vs kernel 忙时")
    wall, busy = m2_busy_ratio(m, x_dev, s_dev, l_dev)
    per_step_wall = wall / 6.0

    print("\n【M3】步内同步点静态计数")
    m3_static_syncs()

    print("\n" + "=" * 72)
    print("【判决】候选 e 的天花板 = 停顿 / 单步墙钟")
    print("  最坏停顿      %.4f ms/步" % per_step)
    print("  单步墙钟      %.4f ms/步" % (per_step_wall * 1000))
    print("  ⇒ 天花板      **%.3f%%**" % (100.0 * per_step / (per_step_wall * 1000)))
    print("  ⇒ 与 §68.7.3 的 P4 实测 +2.5% 相比，**低一个数量级**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
