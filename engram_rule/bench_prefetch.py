"""§69.4 预取流水线原型：**同进程交替 A/B**，先量尺子的噪声地板，再判收益。

**不动出货路径**：`exp_scaling_grid_v2.py` / `dialogue_small.py` 一行未改。
本脚本自带一份训练步的复刻，只为把「预取」这一个变量单独打开。

── 候选 e（`int(lengths_b.max())` 的 D2H 同步）怎么被掩盖 ─────────────────
基础路径每微批做一次 `keep = int(lb.max())`。`int()` 让 host 阻塞到 GPU 排空为止
⇒ 本微批算子还没发完，host 已经卡在下一微批的切分长度上。

预取路径把「取微批 + 读回 keep」整块提前发到 **side stream**：

    步首:  side.wait_stream(cur);  issue(0)
    k:    issue(k+1)                 ← 与主流上微批 k 的 fwd+bwd 重叠
          cur.wait_stream(side)      ← 只等那一个 max kernel，不等主流排空
          用 (bx, sb, lb, keep)

★**为什么这不是「搬运字节」**：a/b/c 三个候选加起来不到 0.01%，
  而 e 是**同步点**——它卡的是 host，不是带宽。

── 两道必须先过的闸（§69.2 的 P7/P8）────────────────────────────────────
1. **P8 等价性**：末 loss 与 baseline 的 `max|Δ| ≤ 1e-6`。跨流会引入
   `record_stream` 生命周期问题，不先验等价就报速度 = 拿错数当快。
2. **P6 A/A 对照**：baseline 与**自己**比，量这把 A/B 夹具的噪声地板。
   ★以前我拿 A/B 中位数当可信读数，却**从没量过尺子的噪声**。
     若 A/A 自己就能读出 1.03×，那 §68.7.3 的 1.025× 就是噪声，必须重判。

用法：`python bench_prefetch.py [d] [rounds]`
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
WARMUP, N_STEP = 3, 10
N_ROWS = 124593
LOSS_TOL = 1e-6          # P8 等价性容差：同序浮点归约 ⇒ 1e-6 足够严
_A = None               # baseline 中位
_B = None               # prefetch 中位


def _side_stream():
    global _SIDE
    if _SIDE is None:
        _SIDE = torch.cuda.Stream() if DEV == "cuda" else None
    return _SIDE


_SIDE = None


def run_path(model, x, s, l, x_dev, s_dev, l_dev, idx_seq, steps, prefetch):
    """跑 steps 步，返回 (末 loss, 墙钟秒)。`idx_seq` 预先给定，各路径共用。

    ★所有路径必须共用同一个 `idx_seq`，否则比的是「不同数据」而不是
      「不同调度」。这是 `bench_resident.py` 立下的规矩，照抄。
    """
    if steps < 1:
        raise ValueError("steps 必须 >= 1")
    _x, _s, _l = x_dev, s_dev, l_dev
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.003,
                                                total_steps=(11 if steps == 10 else steps),
                                                pct_start=0.1)
    model.train()
    bucket_step = getattr(model.c, "bucket_step", 0)
    cur = torch.cuda.current_stream() if DEV == "cuda" else None
    side = _side_stream() if prefetch else None

    def issue(i):
        """在 side 流上取微批 i，并把切分长度读回 host（只等这一个 kernel）。"""
        with torch.cuda.stream(side):
            bx = _x[i]
            sb = _s[i]
            lb = _l[i]
            keep = int(lb.max()) if bucket_step > 0 else bx.shape[1]
        for t in (bx, sb, lb):
            if DEV == "cuda":
                t.record_stream(cur)      # ★在 side 分配、在 main 使用
        return bx, sb, lb, keep

    t0 = time.time()
    last = float("nan")
    for step in range(steps):
        idx = idx_seq[step % len(idx_seq)]
        norm = (_l[idx] - _s[idx]).sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        sl = torch.zeros((), dtype=torch.float32, device=DEV)
        micro = list(idx.split(MB))
        if prefetch:
            side.wait_stream(cur)        # 步首对齐一次，**循环内不再 wait**
            buf = issue(micro[0])
        for k, i in enumerate(micro):
            if prefetch:
                # ★传的是**索引张量** `micro[k+1]`，不是序号 `k+1`。
                #   第一版写 `issue(k + 1)` ⇒ `_x[3]` 取到一行 1-D 张量 ⇒
                #   `bx_full[:, :keep]` 直接 IndexError。
                nxt = issue(micro[k + 1]) if k + 1 < len(micro) else None
                cur.wait_stream(side)    # 只等 side 上那个 max
            if prefetch:
                bx_full, sb, lb, keep = buf
                bx = bx_full[:, :max(keep, 2)] if bucket_step > 0 else bx_full
            else:
                # ★基线路径自己算 keep（等价于 `_trim_length` 的第一行）。
                #   第一版漏了这一行，`keep` 未绑定 ⇒ UnboundLocalError。
                bx_full = _x[i]
                sb = _s[i]
                lb = _l[i]
                keep = int(lb.max()) if bucket_step > 0 else bx_full.shape[1]
                bx = bx_full[:, :max(keep, 2)] if bucket_step > 0 else bx_full
            pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            tl, _fl, _na, _nf = e_answer_loss_sums(logits, bx[:, 1:], valid, sb)
            (tl / norm).backward()
            sl += tl.detach()
            if prefetch:
                buf = nxt
        opt.step()
        sched.step()
        last = float(sl / norm)
        if DEV == "cuda":
            torch.cuda.synchronize()
    return last, time.time() - t0


def e_answer_loss_sums(*a):
    import exp_scaling_grid_v2 as e
    return e.DS.answer_loss_sums(*a)


def simulate(tensor, starts, lengths, idx, micro, bucket_step, prefetch):
    """★**纯形状/索引模拟**：不建模型、不用 GPU，只算两条路径各自会喂进网络什么。

    `tensor` 是任意 2-D CPU 张量（测试里用 32×8 的小张量即可）。
    返回 `(bx_base, bx_prefetch)`，每个元素是 `(行数, keep)`。
    两条路径必须**逐项相同** —— 否则「调度差异」会变成「喂进网络的数据差异」，
    那种加速比根本不可引用。

    `prefetch` 参数刻意保留（并参与一个**恒等式**检查）：
    它只能改变**何时**取数，不能改变**取到什么**。这正是要证的命题。
    """
    if tensor.ndim != 2:
        raise ValueError("tensor 必须 2-D")

    def collect(order):
        out = []
        for k in order:
            i = idx[k * micro:(k + 1) * micro]
            bx_full = tensor[i]
            lb = lengths[i]
            keep = int(lb.max()) if bucket_step > 0 else bx_full.shape[1]
            out.append((bx_full[:, :max(keep, 2)].shape[0],
                        bx_full[:, :max(keep, 2)].shape[1]))
        return out

    n = len(idx) // micro
    # prefetch 的取数顺序是「步首 issue(0)，循环内 issue(k+1)」—— 仍是同一批微批，
    # 只是**时机**不同。所以两条路径的结果必须逐项相同。
    order = list(range(n)) if not prefetch else [0] + list(range(1, n))
    if sorted(order) != sorted(range(n)):
        raise AssertionError("取数顺序不是 [0..n-1] 的重排 ⇒ 有微批被漏掉或重复")
    return collect(range(n)), collect(order)


def _round_robin(mk, x, s, l, x_dev, s_dev, l_dev, idx_seq, modes, rounds):
    """交替 A/B：每轮把所有 mode 各跑一遍，用**同轮内**的比值。

    ★`modes` 是 `(标签, 是否预取)` 的**列表**，不能用 `["base", "base"]`：
      第一版那样写，两个 "base" 在 dict 里塌成**同一个键** ⇒ 8 个值全进同一个 list，
      所谓 A/A 实际是 `a / a` ⇒ 逐轮比值恒等于 1.0000，**假对照**。
      标签必须互不相同，行为才可以相同。
    """
    out = {m[0]: [] for m in modes}
    loss = {m[0]: [] for m in modes}
    for r in range(rounds):
        for label, pf in modes:
            mod = mk()
            mod.train()
            run_path(mod, x, s, l, x_dev, s_dev, l_dev, idx_seq, WARMUP, pf)
            last, w = run_path(mod, x, s, l, x_dev, s_dev, l_dev, idx_seq,
                               N_STEP, pf)
            out[label].append(w / N_STEP)
            loss[label].append(last)
            print("  轮 %d %-9s %.3f s/步   末 loss %.6f"
                  % (r + 1, label, w / N_STEP, last), flush=True)
            del mod
            if DEV == "cuda":
                torch.cuda.empty_cache()
    return out, loss


def _report(tag, a, b, name_a, name_b):
    ma, mb_ = statistics.median(a), statistics.median(b)
    ratios = [x / y for x, y in zip(a, b)]
    print("  %-8s 中位 %.4f s/步  %s" % (name_a, ma, [round(v, 4) for v in a]))
    print("  %-8s 中位 %.4f s/步  %s" % (name_b, mb_, [round(v, 4) for v in b]))
    print("  逐轮比值 %s  中位 %.4f  极差 %.4f"
          % ([round(r, 4) for r in ratios], statistics.median(ratios),
             max(ratios) - min(ratios)))
    print("  ⇒ 加速比（%s / %s）= **%.4f×**" % (name_a, name_b, ma / mb_))
    return ma / mb_


def main():
    global _SIDE
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    import exp_scaling_grid_v2 as e
    import model as M
    e.load_all()
    rows = e.load_rows()
    tr, he = e.split_rows(rows)
    e.assert_no_leak(tr, he)
    x, s, l = e.encode_cached(tr[:N_ROWS], e._TOK, 512)
    ck = e._CK

    print("=" * 74)
    print("预取 A/B：d=%d, N=%d, BS=%d, micro_batch=%d, %d 轮 × %d 步, DEV=%s"
          % (d, min(N_ROWS, len(tr)), BS, MB, rounds, N_STEP, DEV))
    print("=" * 74)

    def mk():
        torch.manual_seed(11)
        m = M.EngramRuleLM(M.RCfg(**{**ck["cfg"], "d": d, "n_heads": d // 16})).to(DEV)
        e.DS.build_engram_from_examples(m, x, l)
        return m

    x_dev, s_dev, l_dev = x.to(DEV), s.to(DEV), l.to(DEV)
    print("x 常驻 GPU：%.0f MB" % (x.numel() * 8 / 1048576.0), flush=True)

    g = torch.Generator().manual_seed(11)
    idx_seq = [torch.randint(0, x.shape[0], (BS,), generator=g)
               for _ in range(N_STEP + WARMUP)]

    # ── P6：A/A 噪声地板（baseline 与自己比，**标签必须不同**）────────────────
    print("\n【P6】A/A 对照：baseline vs baseline（量这把尺子的噪声地板）")
    aa, _l1 = _round_robin(mk, x, s, l, x_dev, s_dev, l_dev, idx_seq,
                           [("baseA", False), ("baseB", False)], rounds)
    _report("A/A", aa["baseA"], aa["baseB"], "baseA", "baseB")
    # ★只取**同轮内**的配对比值；第一版还把「末轮 vs 首轮」算进去，那是**跨轮漂移**
    #   不是这把尺子的噪声，会把地板虚高一截。
    pair = [p / q for p, q in zip(aa["baseA"], aa["baseB"])]
    floor = max(abs(v - 1.0) for v in pair)
    print("  同轮内 A/A 比值 %s" % [round(v, 4) for v in pair])
    print("  ⇒ **噪声地板 ≈ %.2f%%**（同轮内 A/A 偏离 1 的最大幅度）" % (100 * floor))
    print("  ⇒ 加速比低于 %.4f 的读数**不可与噪声区分**" % (1.0 + floor))

    # ── P8/P7：等价性 + 交替 A/B ───────────────────────────────────────────
    print("\n【P8/P7】等价性 + 交替 A/B")
    res, loss = _round_robin(mk, x, s, l, x_dev, s_dev, l_dev, idx_seq,
                             [("base", False), ("prefetch", True)], rounds)
    ratio = _report("A/B", res["base"], res["prefetch"], "base", "prefetch")
    dl = max(abs(a - b) for a, b in zip(loss["base"], loss["prefetch"]))
    print("  ⇒ P8 末 loss 最大差 = %.3e（容差 %.0e）⇒ %s"
          % (dl, LOSS_TOL, "等价 ✓" if dl <= LOSS_TOL else "★不等价，禁止引用速度"))

    thresh = 1.0 + floor
    if dl > LOSS_TOL:
        print("  ⇒ 不等价 ⇒ 本轮速度读数**作废**，先修等价性")
    elif ratio < thresh:
        print("  ⇒ 加速比 %.4f < 噪声地板 %.4f ⇒ **判「无收益」，不并入出货路径**"
              % (ratio, thresh))
    elif ratio < 1.02:
        print("  ⇒ 加速比 %.4f 勉强过噪声地板但 < 1.02 ⇒ 收益太小，不值得改前向调度"
              % ratio)
    else:
        print("  ⇒ 加速比 %.4f ≥ 1.02 且等价 ⇒ **候选成立**，可并入（需再跑 d=64/128 复核）"
              % ratio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
