"""§68.7：修复点在**真实训练循环**上的交替 A/B + 继续排查剩余同步点。

**为什么必须交替 A/B**：本机 GPU 空闲利用率仍 99%（AGENTS 规则 22），
同一档配置跨轮测的绝对值可漂 18%（§68.5 实测 0.244 → 0.287）。
只有**同进程内交替、用同轮比值**才能排除。

**为什么不能用 profiler 的同步计数**：负向对照已证它量不到 `int(张量)`
（`bench_sync_points.py` 的 [FAIL]：恢复断言后计数不变）。`int(张量)` 走
PyTorch 内部 `.item()`，不一定在 profiler 留 CUDA API 条目。

**本脚本做两件事**：
  A. 改前(dist/model.py) vs 改后(model.py) 在**真实训练步**上交替 A/B
  B. 静态扫描两个文件里**所有**可能的 D2H 同步点，列出未修的
"""
import importlib.util
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
WARMUP, N_STEP, ROUNDS = 3, 10, 3


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def run_train(mod, e, x, s, l, idx_seq, steps, d):
    """与 train_steps 同口径。total_steps=10 会让 OneCycleLR 除零 ⇒ 抬到 11。"""
    torch.manual_seed(11)
    m = mod.EngramRuleLM(mod.RCfg(**{**e._CK["cfg"], "d": d,
                                      "n_heads": d // 16})).to(DEV)
    e.DS.build_engram_from_examples(m, x, l)
    opt = torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.003,
                                                total_steps=(11 if steps == 10 else steps),
                                                pct_start=0.1)
    m.train()
    bucket_step = getattr(m.c, "bucket_step", 0)
    t0 = time.time()
    last = float("nan")
    for step in range(steps):
        idx = idx_seq[step % len(idx_seq)]
        norm = (l[idx] - s[idx]).sum().clamp_min(1).to(DEV)
        opt.zero_grad(set_to_none=True)
        sl = torch.zeros((), dtype=torch.float32, device=DEV)
        for i in idx.split(MB):
            bx = x[i].to(DEV)
            sb = s[i].to(DEV)
            lb = l[i].to(DEV)
            bx = e.DS._trim_length(bx, lb) if bucket_step > 0 else bx
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


# ── B. 静态扫描剩余同步点 ────────────────────────────────────────────────
SYNC_PATTERNS = (
    "int(", "float(", "bool(", ".item()", ".cpu()", ".tolist()",
    ".numpy()", "nonzero", "masked_select", "index_select",
)


def scan_sync_points(path):
    """列出**热路径**里的 D2H 同步候选（排除注释行与 docstring）。"""
    hits = []
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    in_doc = False
    for i, ln in enumerate(lines, 1):
        t = ln.strip()
        if not t or t.startswith("#"):
            continue
        # 粗略跟踪三引号 docstring
        q = t.count('"""')
        if in_doc:
            if q % 2 == 1:
                in_doc = False
            continue
        if q % 2 == 1:
            in_doc = True
            continue
        for p in SYNC_PATTERNS:
            if p in t:
                hits.append((i, p, t[:88]))
                break
    return hits


def main():
    import exp_scaling_grid_v2 as e
    e.load_all()
    rows = e.load_rows()
    tr, he = e.split_rows(rows)
    e.assert_no_leak(tr, he)
    x, s, l = e.encode_cached(tr[:8192], e._TOK, 512)
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 64

    print("=" * 76)
    print("B. 静态扫描：剩余 D2H 同步候选（已排注释与 docstring）")
    print("=" * 76)
    for tag, path in (("dist/model.py（改前）", os.path.join(HERE, "dist", "model.py")),
                      ("model.py（改后）", os.path.join(HERE, "model.py"))):
        hits = scan_sync_points(path)
        print("  %s：%d 处" % (tag, len(hits)))
        for i, p, t in hits:
            print("    %4d  %-14s %s" % (i, p, t))
        print()

    print("=" * 76)
    print("A. 真实训练步上的交替 A/B（d=%d，%d 轮 × %d 步）"
          % (d, ROUNDS, N_STEP))
    print("=" * 76)
    new = load(os.path.join(HERE, "model.py"), "m_new")
    old = load(os.path.join(HERE, "dist", "model.py"), "m_old")

    g = torch.Generator().manual_seed(11)
    idx_seq = [torch.randint(0, x.shape[0], (BS,), generator=g)
               for _ in range(N_STEP + WARMUP)]

    base_t, new_t, base_l, new_l = [], [], [], []
    for r in range(ROUNDS):
        for tag, mod, tt, ll in (("改前", old, base_t, base_l),
                                 ("改后", new, new_t, new_l)):
            last, _w = run_train(mod, e, x, s, l, idx_seq, WARMUP, d)
            last, w = run_train(mod, e, x, s, l, idx_seq, N_STEP, d)
            tt.append(w / N_STEP); ll.append(last)
            print("  轮 %d %s  %.4f s/步   末 loss %.6f" % (r + 1, tag, w / N_STEP, last),
                  flush=True)
            torch.cuda.empty_cache()

    b, c = statistics.median(base_t), statistics.median(new_t)
    print()
    print("  改前 中位 %.4f s/步  %s" % (b, [round(v, 4) for v in base_t]))
    print("  改后 中位 %.4f s/步  %s" % (c, [round(v, 4) for v in new_t]))
    print("  ⇒ 加速比 %.4fx（%+.2f%%）  %s"
          % (b / c, 100 * (b / c - 1),
             "PASS >1.02" if b / c > 1.02 else
             ("★更慢" if b / c < 0.98 else "无实质差异")))
    print("  ⇒ 末 loss 最大差 %.3e" % max(abs(p - q) for p, q in zip(base_l, new_l)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
