"""★torch.compile 的 A/B：**同进程、交替测**，排除机器负载漂移。

**为什么必须交替**：本机桌面合成进程使 GPU 空闲利用率仍有 99%（AGENTS 规则 22）。
前后两轮分开测的读数差 18%（d=64 的 0.244 vs 0.287）—— 那不是优化效果，是抢占。
⇒ 交替 A/B-A/B-A/B，用**同档内的比值**，跨档绝对值不作结论。

**口径**：与 `train_steps` 逐字相同（trim + answer_loss_sums + 微批 + AdamW + OneCycle）。
只改一处：`model = torch.compile(model)`。

**判据（预登记）**：
- T1 加速比 = 中位(旧) / 中位(新)，>1.05 才算有实质收益
- T2 数值：编译会改归约顺序 ⇒ 末位可能变。**同时报 max|Δ|**，让桃自己决定能否接受
      （规则 14：数值等价要给机器精度容差；但 compile 是算子融合，不一定满足）
- 负向对照：把 `mode` 设成明显无效的值（如 compile 后立即销毁），比值必须回到 ~1

**不做出货路径修改**：`torch.compile` 只包在实验脚本里，`model.py` 一行不改。
"""
import os
import statistics
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BACKEND = os.environ.get("COMPILE_BACKEND", "cudagraphs")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 3
N_STEP = 8
ROUNDS = 3


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
    mb = 64

    def make():
        torch.manual_seed(11)
        m = M.EngramRuleLM(M.RCfg(**{**ck["cfg"], "d": d,
                                      "n_heads": d // 16})).to(DEV)
        e.DS.build_engram_from_examples(m, x, l)
        m.train()
        return m

    def run_once(m, n_step):
        """返回 (末 loss, 墙钟秒)。与 train_steps 同口径。"""
        opt = torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.003,
                                                    total_steps=n_step, pct_start=0.1)
        g = torch.Generator().manual_seed(11)
        bs_ = getattr(m.c, "bucket_step", 0)
        t0 = time.time()
        last = float("nan")
        for step in range(n_step):
            idx = torch.randint(0, x.shape[0], (256,), generator=g)
            norm = (l[idx] - s[idx]).sum().clamp_min(1).to(DEV)
            opt.zero_grad(set_to_none=True)
            sl = torch.zeros((), dtype=torch.float32, device=DEV)
            for i in idx.split(mb):
                bxf = x[i].to(DEV); sb = s[i].to(DEV); lb = l[i].to(DEV)
                bx = bxf[:, :max(int(lb.max()), 2)] if bs_ > 0 else bxf
                pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
                valid = pos < lb.view(-1, 1)
                logits = m(bx)[:, :-1, :]
                tl, _fl, _na, _nf = e.DS.answer_loss_sums(logits, bx[:, 1:], valid, sb)
                (tl / norm).backward()
                sl += tl.detach()
            opt.step(); sched.step()
            last = float(sl / norm)
            if DEV == "cuda":
                torch.cuda.synchronize()
        return last, time.time() - t0

    # ---- 数值对拍（先确认 compile 前后差多少）----
    print("=" * 70)
    print("数值对拍：compile 前 vs compile 后，同一 batch 的 logits")
    print("=" * 70)
    m_ref = make(); m_ref.eval()
    xb = x[:4, :180].to(DEV)
    with torch.no_grad():
        y_ref = m_ref(xb)
    del m_ref; torch.cuda.empty_cache()

    m_c = make(); m_c.eval()
    try:
        m_cc = torch.compile(m_c, backend=BACKEND)
        with torch.no_grad():
            y_c = m_cc(xb)
        mx = float((y_ref - y_c).abs().max())
        print("  max|Δ| = %.3e   %s" % (mx, "逐位相同" if mx == 0 else "末位有差"))
    except Exception as ex:
        mx = None
        print("  compile 失败: %s" % str(ex)[:200])
    del m_c, m_cc
    torch.cuda.empty_cache()

    # ---- 交替 A/B ----
    print()
    print("=" * 70)
    print("交替 A/B：%d 轮 × %d 步（预热 %d 步不计时）" % (ROUNDS, N_STEP, WARMUP))
    print("=" * 70)
    base_t, comp_t = [], []
    for r in range(ROUNDS):
        # A：未编译
        m = make()
        _l, _w = run_once(m, WARMUP)
        _l, w = run_once(m, N_STEP)
        base_t.append(w / N_STEP)
        del m; torch.cuda.empty_cache()
        # B：编译（每轮新建，避免复用编译缓存掩盖真实编译开销）
        m = make()
        try:
            m = torch.compile(m, backend=BACKEND)
        except Exception as ex:
            print("  compile 失败: %s" % str(ex)[:150])
            del m; torch.cuda.empty_cache()
            break
        _l, _w = run_once(m, WARMUP)
        _l, w = run_once(m, N_STEP)
        comp_t.append(w / N_STEP)
        del m
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        print("  轮 %d: 未编译 %.3f s/步   编译 %.3f s/步"
              % (r + 1, base_t[-1], comp_t[-1]), flush=True)

    if not base_t or not comp_t:
        print("未能完成 A/B")
        return 1
    b, c = statistics.median(base_t), statistics.median(comp_t)
    print()
    print("  未编译  中位 %.3f s/步  （%d 轮：%s）"
          % (b, len(base_t), [round(v, 3) for v in base_t]))
    print("  编译    中位 %.3f s/步  （%d 轮：%s）"
          % (c, len(comp_t), [round(v, 3) for v in comp_t]))
    print("  ⇒ 加速比 = %.3f×   %s"
          % (b / c, "PASS（>1.05）" if b / c > 1.05 else "★收益不足"))
    if mx is not None:
        print("  ⇒ 数值 max|Δ| = %.3e" % mx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
