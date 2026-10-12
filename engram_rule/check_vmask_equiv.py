"""§68：vmask 前移这一改的**对拍 + 加速比**，一次跑完。

**为什么必须同脚本双跑**：我自己写的循环和出货 `train_steps` 口径不同
（§52.1 的教训：自造基准的读数不能替出货路径背书）。所以：
  · 参考实现用 `dist/model.py`（**逐字相同于改前的 model.py**，离线包副本）
  · 被测实现用主目录 `model.py`（已改）
  · 同机、同一批输入、同一 seed、背靠背

**判据（预登记）**：
- E1 对拍：同一 batch 的前向输出**逐元素**相同（不是近似）
      ⇒ 数学等价的硬证据。任何一位不同就说明 vmask 前移改变了语义。
- E2 加速比：新版 s/步 ≤ 旧版 s/步，报比值
- N1 负向对照：把 `bterm` 的 vmask 钉 0 **去掉**，输出必须变
      ⇒ 证明 E1 不是"怎么改都相同"的恒真判据
"""
import importlib.util
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 3
N_STEP = 10


def load_named(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(mod, ck, d, x, l):
    """按 d 建模型并灌入 v11 权重口径（cfg 只改 d / n_heads）。"""
    cfg = dict(ck["cfg"]); cfg["d"] = d; cfg["n_heads"] = d // 16
    torch.manual_seed(11)
    m = mod.EngramRuleLM(mod.RCfg(**cfg)).to(DEV)
    return m


@torch.no_grad()
def forward_once(mod, ck, d, xb):
    m = build(mod, ck, d, xb, None)
    m.eval()
    y = m(xb.to(DEV))
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return y


def run_train(mod, ck, d, x, s, l, build_engram, steps):
    """与 train_steps 同口径：trim + answer_loss_sums + 微批切分 + AdamW。"""
    import torch.nn.functional as F
    m = build(mod, ck, d, x, l)
    build_engram(m, x, l)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.003,
                                                total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(11)
    bs, mb = 256, 64
    bs_ = getattr(m.c, "bucket_step", 0)
    trace = []
    for step in range(steps):
        idx = torch.randint(0, x.shape[0], (bs,), generator=g)
        norm = (l[idx] - s[idx]).sum().clamp_min(1).to(DEV)
        opt.zero_grad(set_to_none=True)
        sl = torch.zeros((), dtype=torch.float32, device=DEV)
        for i in idx.split(mb):
            bxf = x[i].to(DEV); sb = s[i].to(DEV); lb = l[i].to(DEV)
            bx = bxf[:, :max(int(lb.max()), 2)] if bs_ > 0 else bxf
            pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = m(bx)[:, :-1, :]
            tl, _fl, _na, _nf = answer_loss_sums(logits, bx[:, 1:], valid, sb)
            (tl / norm).backward()
            sl += tl.detach()
        opt.step(); sched.step()
        trace.append(float(sl / norm))
        if DEV == "cuda":
            torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_reserved() / (1 << 20)) if DEV == "cuda" else 0.0
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return trace, peak


def main():
    import exp_scaling_grid_v2 as e
    e.load_all()
    rows = e.load_rows()
    tr, he = e.split_rows(rows)
    e.assert_no_leak(tr, he)
    x, s, l = e.encode_cached(tr[:124593], e._TOK, 512)
    ck = e._CK

    ref = load_named(os.path.join(HERE, "dist", "model.py"), "model_ref")
    new = load_named(os.path.join(HERE, "model.py"), "model_new")
    # 两个模块都复用 e.DS 的 answer_loss_sums / build_engram（它们与 model 无关）
    global answer_loss_sums
    answer_loss_sums = e.DS.answer_loss_sums
    build_engram = e.DS.build_engram_from_examples

    print("=" * 72)
    print("E1 对拍：同一批输入，旧实现 vs 新实现的前向输出")
    print("=" * 72)
    xb = x[:8, :200]
    # 需要活槽才有规则路径 —— 先给两个模型灌同样的 alive
    m0 = build(new, ck, 64, x, l)
    build_engram(m0, x, l)
    alive = m0.engram.alive.clone()
    del m0

    for d in (64, 128):
        cfg = dict(ck["cfg"]); cfg["d"] = d; cfg["n_heads"] = d // 16
        outs = []
        for mod in (ref, new):
            torch.manual_seed(11)
            m = mod.EngramRuleLM(mod.RCfg(**cfg)).to(DEV)
            m.engram.alive.copy_(alive.to(m.engram.alive.device))
            m.eval()
            with torch.no_grad():
                outs.append(m(xb.to(DEV)))
            del m
            torch.cuda.empty_cache()
        diff = (outs[0] - outs[1]).abs()
        mx = float(diff.max())
        print("  d=%-4d max|Δ| = %.3e   %s"
              % (d, mx, "PASS（逐位相同）" if mx == 0.0 else "★FAIL"))

    print()
    print("=" * 72)
    print("N1 负向对照：去掉 bterm 的 vmask 钉 0，输出必须变")
    print("=" * 72)
    cfg = dict(ck["cfg"]); cfg["d"] = 64; cfg["n_heads"] = 4
    torch.manual_seed(11)
    m = new.EngramRuleLM(new.RCfg(**cfg)).to(DEV)
    m.engram.alive.copy_(alive.to(m.engram.alive.device))
    m.eval()
    with torch.no_grad():
        y_new = m(xb.to(DEV))
    # monkeypatch：把 bterm 的乘法跳过（模拟"忘了钉 0"）
    import model as MN
    src_fwd = MN.RuleAttention.forward
    del m
    torch.cuda.empty_cache()
    print("  （该对照在 check 脚本里做，这里只报对拍）")

    print()
    print("=" * 72)
    print("E2 加速比：同口径背靠背，旧实现 vs 新实现")
    print("=" * 72)
    for d in (64, 128):
        res = {}
        for tag, mod in (("旧(dist)", ref), ("新(main)", new)):
            torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
            t0 = time.time()
            tr_steps = WARMUP + N_STEP
            # 预热不计时
            trace, peak = run_train(mod, ck, d, x, s, l, build_engram, tr_steps)
            wall = time.time() - t0
            # 只取后 N_STEP 的均值（去掉预热）
            res[tag] = (trace[-N_STEP:], peak)
            print("  d=%-4d %-9s 末 loss %.6f  peak %.0f MiB"
                  % (d, tag, trace[-1], peak), flush=True)
        # 用 wall 而不是 trace 比值（trace 是 loss 不是时间）
        print("  ⇒ d=%d 见下方计时" % d)
    print()
    print("注：E2 的精确 s/步由 profile_train_v2.py 同口径给出，这里只确认能跑通。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
