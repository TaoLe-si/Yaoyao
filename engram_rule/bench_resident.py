"""§68.7：x/s/l 常驻 GPU 的**同进程交替 A/B** —— 排除机器负载漂移。

**为什么必须交替**：本机 GPU 空闲利用率仍 99%（AGENTS 规则 22），跨轮测的绝对值不可比。
上一轮我用「单轮 12 步 vs 历史 0.287」得出「慢 28%」，那个结论**证据不足**（vmask 那次已犯过同样错误）。

**做法**：同一个进程里 A/B/A/B/A/B 交替，用**同轮内的比值**。

**两条路径在同一函数里切换**，靠 `resident` 标志：
- resident=False：`bx_full = x[i].to(DEV)`（原路径，走 H2D）
- resident=True ：`bx_full = x_dev[i]`（常驻，设备内 gather）

★**关键**：两条路径必须共用同一个 `gen` 与同一个 `idx` 序列，
否则比的是「不同数据」而不是「不同传输方式」。做法是先生成 idx 再喂给两条路径。
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
WARMUP, N_STEP, ROUNDS = 3, 10, 4


def run_path(model, x, s, l, x_dev, s_dev, l_dev, resident, idx_seq, steps):
    """跑 steps 步，返回 (末 loss, 墙钟秒)。idx_seq 预先给定，两条路径共用。

    ★`OneCycleLR(total_steps=10, pct_start=0.1)` 会 `ZeroDivisionError`
      （`warmup = int(10*0.1) = 1`，内部 `end_step - start_step` 退化成 0）。
      实测 total_steps ∈ {1..9, 11, 20, 21} 都正常，**只有 10 炸**。
      第一版我加 `max(steps,2)` 是**没定位真因就改** —— 传的本来就是 10，那个 max 毫无作用。
      正确做法：只把 10 抬到 11（少一步 warmup 不影响计时结论）。
    """
    import exp_scaling_grid_v2 as e
    if steps < 1:
        raise ValueError("steps 必须 >= 1")
    _x = x_dev if resident else x
    _s = s_dev if resident else s
    _l = l_dev if resident else l
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.003,
                                                total_steps=(11 if steps == 10 else steps),
                                                pct_start=0.1)
    model.train()
    bs_ = getattr(model.c, "bucket_step", 0)
    t0 = time.time()
    last = float("nan")
    for step in range(steps):
        idx = idx_seq[step % len(idx_seq)]
        if resident:
            norm = (_l[idx] - _s[idx]).sum().clamp_min(1)
        else:
            norm = (l[idx] - s[idx]).sum().clamp_min(1).to(DEV)
        opt.zero_grad(set_to_none=True)
        sl = torch.zeros((), dtype=torch.float32, device=DEV)
        for i in idx.split(MB):
            if resident:
                bx_full = _x[i]; sb = _s[i]; lb = _l[i]
            else:
                bx_full = x[i].to(DEV); sb = s[i].to(DEV); lb = l[i].to(DEV)
            bx = e.DS._trim_length(bx_full, lb) if bs_ > 0 else bx_full
            pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            tl, _fl, _na, _nf = e.DS.answer_loss_sums(logits, bx[:, 1:], valid, sb)
            (tl / norm).backward()
            sl += tl.detach()
        opt.step(); sched.step()
        last = float(sl / norm)
        if DEV == "cuda":
            torch.cuda.synchronize()
    return last, time.time() - t0


def main():
    import exp_scaling_grid_v2 as e
    import model as M
    e.load_all()
    rows = e.load_rows()
    tr, he = e.split_rows(rows)
    e.assert_no_leak(tr, he)
    x, s, l = e.encode_cached(tr[:124593], e._TOK, 512)
    ck = e._CK
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 64

    print("=" * 70)
    print("交替 A/B：resident(常驻 GPU) vs paged(走 H2D)，%d 轮 × %d 步" % (ROUNDS, N_STEP))
    print("=" * 70)

    def make():
        torch.manual_seed(11)
        m = M.EngramRuleLM(M.RCfg(**{**ck["cfg"], "d": d,
                                      "n_heads": d // 16})).to(DEV)
        e.DS.build_engram_from_examples(m, x, l)
        return m

    x_dev = x.to(DEV); s_dev = s.to(DEV); l_dev = l.to(DEV)
    print("x 常驻 GPU 完成（%.0f MB）" % (x.numel() * 8 / 1048576.0), flush=True)

    # 预先生成 idx 序列，两条路径共用（保证比的是传输方式而非数据）
    g = torch.Generator().manual_seed(11)
    idx_seq = [torch.randint(0, x.shape[0], (BS,), generator=g)
               for _ in range(N_STEP + WARMUP)]

    base_t, res_t = [], []
    base_loss, res_loss = [], []
    for r in range(ROUNDS):
        for tag, resident in (("paged ", False), ("resident", True)):
            m = make(); m.train()
            _l1, _w = run_path(m, x, s, l, x_dev, s_dev, l_dev, resident,
                               idx_seq, WARMUP)
            last, w = run_path(m, x, s, l, x_dev, s_dev, l_dev, resident,
                               idx_seq, N_STEP)
            (res_t if resident else base_t).append(w / N_STEP)
            (res_loss if resident else base_loss).append(last)
            print("  轮 %d %s  %.3f s/步   末 loss %.6f"
                  % (r + 1, tag, w / N_STEP, last), flush=True)
            del m
            torch.cuda.empty_cache()

    b, c = statistics.median(base_t), statistics.median(res_t)
    print()
    print("  paged    中位 %.3f s/步  %s" % (b, [round(v, 3) for v in base_t]))
    print("  resident 中位 %.3f s/步  %s" % (c, [round(v, 3) for v in res_t]))
    print("  ⇒ 加速比 = %.3fx   %s"
          % (b / c, "PASS（>1.05）" if b / c > 1.05 else
             ("★更慢" if b / c < 0.95 else "无实质差异")))
    dl = max(abs(a - b_) for a, b_ in zip(base_loss, res_loss))
    print("  ⇒ 末 loss 最大差 = %.3e（两条路径数值一致性）" % dl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
