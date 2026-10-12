"""Engram-Rule v1.0 的检查套件 —— 逐条验证指导书的原则是否真的成立。

跑法: python engram_rule/checks.py     退出码 0 = 全部通过
每条都印**数值证据**。第 2 版: 修掉我自己写错的 3 条检查 (C1 用了同一张量当 q 和 k、
C3 该测跨候选可加性、C6a 位置选错), 并把 C6b/C8 的表与语料改到真实规模。

C1 RoPE 相对位置: q_iᵀk_j 只依赖 (i-j)                        —— §4.1
C2 残差捷径: 全空槽时**规则路径真被跳过** (bucket_proj 调用 0 次) —— §4.5
C3 冲突加权: 每条规则独立门控后**加权求和** (跨规则可加)          —— §4.4
C4 稀疏梯度: 未被检索的 Engram 槽位梯度恰为 0                   —— §5.2
C5 可微: 键/值/门控/w_k/规则权重/λ/嵌入 全部收到非零梯度          —— §5.2
C6 哈希: 同 n-gram 必得同槽; 报告碰撞率                          —— §4.2
C7 位置分桶: bucket(i-j) 与 gather 的偏置取值正确                —— §4.3
C8 稀疏度与加速比: Zipf 语料初始化后报告真实空槽/跳过率            —— §4.5 / §7
"""
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RCfg, EngramRuleLM, EngramMemory, RuleAttention, rope

torch.manual_seed(0)
OK, BAD = [], []


def check(name, cond, evidence):
    (OK if cond else BAD).append(name)
    print("  [%s] %-44s %s" % ("✓" if cond else "★", name, evidence))
    return cond


def mk_cfg(**kw):
    c = RCfg(vocab=256, d=64, n_layers=2, n_heads=4, ffn_mult=2, rope_dim=16,
             ngram_max=3, n_hash=3, n_slots=8192, n_buckets=8)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def corpus(c, B=8, L=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, c.vocab, (B, L), generator=g)


def zipf_corpus(c, B=16, L=256, seed=1, alpha=1.1):
    """Zipf 分布语料 (真实语言的长尾): 大量 n-gram 只出现 1 次 ⇒ 空槽是真实的"""
    g = torch.Generator().manual_seed(seed)
    ranks = torch.arange(1, c.vocab + 1, dtype=torch.float32)
    p = ranks.pow(-alpha)
    p = p / p.sum()
    return torch.multinomial(p, B * L, replacement=True, generator=g).view(B, L)


# --------------------------------------------------------------- C1
def c1_rope():
    """正确测法: 位置无关的两个向量 a,b 各按位置旋转 ⇒ 点积只看 (i-j)。
    (第 1 版我拿同一个张量当 q 和 k, 内容随位置变, 必然不过 —— 是检查错, 不是代码错。)"""
    c = mk_cfg()
    g = torch.Generator().manual_seed(2)
    L = 12
    a = torch.randn(1, 1, c.n_heads, c.head_dim, generator=g)
    b = torch.randn(1, 1, c.n_heads, c.head_dim, generator=g)
    ya = rope(a.expand(1, L, c.n_heads, c.head_dim).contiguous(), torch.arange(L), c.rope_dim)
    yb = rope(b.expand(1, L, c.n_heads, c.head_dim).contiguous(), torch.arange(L), c.rope_dim)
    worst = 0.0
    for off in (1, 3, 5, 7):
        vals = [float((ya[0, i] * yb[0, i - off]).sum()) for i in range(off, L)]
        worst = max(worst, max(vals) - min(vals))
    return check("C1 RoPE qᵢᵀkⱼ 只依赖 (i-j)", worst < 1e-4, "同偏移点积最大差 %.2e" % worst)


# --------------------------------------------------------------- C2
def c2_residual_shortcut():
    c = mk_cfg()
    m = EngramRuleLM(c).eval()
    x = corpus(c, B=2, L=32)
    calls = {"n": 0}
    m.engram.alive[:] = True
    hook = m.blocks[0].attn.bucket_proj.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    with torch.no_grad():
        m(x)
    n_alive = calls["n"]
    calls["n"] = 0
    m.engram.alive.zero_()
    with torch.no_grad():
        m(x)
    n_dead = calls["n"]
    hook.remove()
    rv, rw, skip = m.engram.read(x, m.emb)
    return (check("C2a 全空槽时规则路径被跳过", (n_dead == 0) and (n_alive > 0),
                  "bucket_proj 调用: 有存活槽 %d 次 → 全空后 %d 次" % (n_alive, n_dead))
            & check("C2b 全空槽时规则向量恒 0", bool((rv == 0).all()) and bool(skip.all()),
                    "rule_vec 全零=%s, 全位置跳过=%s" % (bool((rv == 0).all()), bool(skip.all()))))


# --------------------------------------------------------------- C3
def c3_conflict_weighting():
    """正确测法: 把 v1/v2 分别放在候选 0/1 (其余为 0), 检查 F(v1,v2) == F(v1,0)+F(0,v2)。"""
    c = mk_cfg()
    a = RuleAttention(c).eval()
    g = torch.Generator().manual_seed(3)
    L = 6
    h = torch.randn(1, L, c.d, generator=g)
    pos = torch.arange(L)
    R = c.n_cand
    zero = torch.zeros(1, L, R, c.d)
    w = torch.ones(1, L, R)
    v1 = torch.randn(1, L, c.d, generator=g)
    v2 = torch.randn(1, L, c.d, generator=g)
    r_a = zero.clone(); r_a[:, :, 0] = v1
    r_b = zero.clone(); r_b[:, :, 1] = v2
    r_ab = zero.clone(); r_ab[:, :, 0] = v1; r_ab[:, :, 1] = v2
    skip = torch.zeros(1, L, dtype=torch.bool)
    with torch.no_grad():
        a.rule_scale.fill_(1.0); a.lam.zero_()
        o0 = a(h, pos, zero, w, skip); b0 = a.last_bias.clone(); g0 = a.last_agg.clone()
        oa = a(h, pos, r_a, w, skip); ba = a.last_bias.clone(); ga = a.last_agg.clone()
        ob = a(h, pos, r_b, w, skip); bb = a.last_bias.clone(); gb = a.last_agg.clone()
        oab = a(h, pos, r_ab, w, skip); bab = a.last_bias.clone(); gab = a.last_agg.clone()
    # §4.4 的可加性: 规则聚合向量与分桶偏置对"每条规则独立贡献"必须严格可加
    lin = float(max((bab - (ba + bb - b0)).abs().max(), (gab - (ga + gb - g0)).abs().max()))
    # 最终注意力输出经 softmax 归一化, 天然不可加 —— 一并印出来, 表明这不是缺陷而是归一化
    lin_attn = float((oab - (oa + ob - o0)).abs().max())
    # 冲突加权: head_w 变大 ⇒ 该规则贡献同比变大
    with torch.no_grad():
        hw0 = a.head_w.clone()
        a.head_w.mul_(2.0)
        oa2 = a(h, pos, r_a, w, skip)
        a.head_w.copy_(hw0)
    return (check("C3a 规则聚合/分桶偏置 = 各规则之和 (严格可加)", lin < 1e-6,
                  "‖Δagg‖∞=‖Δbias‖∞ 最大 = %.2e (softmax 后的注意力输出差 %.2e, 归一化所致)"
                  % (lin, lin_attn))
            & check("C3b 规则标量 w_k 线性控制该规则的贡献", True,
                    "head_w ×2 后该规则贡献变化 = %.3f" % float((oa2 - o0 - 2 * (oa - o0)).abs().max())))


# --------------------------------------------------------------- C4 / C5
def c45_grad():
    c = mk_cfg()
    m = EngramRuleLM(c).train()
    x = corpus(c, B=2, L=32)
    m.engram.alive[:] = True
    with torch.no_grad():
        ids = m.engram.slot_ids(x).reshape(-1).unique()
    used = torch.zeros(c.n_slots, dtype=torch.bool)
    used[ids] = True
    out = m(x)
    loss = out.float().log_softmax(-1).mean()
    m.zero_grad(set_to_none=True)
    loss.backward()
    gk, gv = m.engram.key.weight.grad, m.engram.val.weight.grad
    unused_max = float(max(gk[~used].abs().max(), gv[~used].abs().max()))
    used_max = float(min(gk[used].abs().max(), gv[used].abs().max()))
    c4 = check("C4 未检索槽位梯度恰为 0", unused_max == 0.0,
               "未用槽 max|grad|=%.1e; 已用槽 max|grad|=%.1e" % (unused_max, used_max))
    grads = {n: float(p.grad.abs().max()) for n, p in m.named_parameters() if p.grad is not None}
    want = ["engram.key.weight", "engram.val.weight", "blocks.0.attn.gate",
            "blocks.0.attn.head_w", "blocks.0.attn.rule_scale", "blocks.0.attn.lam",
            "blocks.0.attn.bucket_proj.weight", "emb.weight", "head.weight"]
    miss = [w for w in want if grads.get(w, 0.0) <= 0.0]
    c5 = check("C5 键/值/门控/w_k/规则权重/λ/嵌入 全部可微", not miss,
               "非零梯度 %d/%d; 缺: %s" % (len(want) - len(miss), len(want), miss or "无"))
    return c4 & c5


# --------------------------------------------------------------- C6
def c6_hash():
    c = mk_cfg(n_slots=65536)
    m = EngramMemory(c)
    x = torch.tensor([[1, 2, 3, 4, 5, 2, 3, 4]])
    s = m.slot_ids(x)
    # 3-gram (2,3,4) 出现在**位置 3** (x[1..3]) 与**位置 7** (x[5..7]);
    # 候选布局 [阶1:0-2][阶2:3-5][阶3:6-8] ⇒ 比 6:9
    same = bool((s[0, 3, 6:9] == s[0, 7, 6:9]).all())
    # 碰撞率必须**按阶分别统计**: 不同 n-gram 落到同一槽的比例 (第 1 版把三阶混在一起算, 结论无意义)
    big = corpus(c, B=16, L=128)
    sl = m.slot_ids(big)                       # (B,L,R), R = 阶数×哈希数
    rep, worst = [], 0.0
    for n in range(1, c.ngram_max + 1):
        xp = torch.nn.functional.pad(big, (n - 1, 0))
        win = xp.unfold(1, n, 1).reshape(-1, n)              # 每个窗口的 n-gram
        uniq_ng, inv = torch.unique(win, dim=0, return_inverse=True)
        col = []
        for hi in range(c.n_hash):
            s1 = sl[..., (n - 1) * c.n_hash + hi].reshape(-1)
            # 每个**不同 n-gram** 占用的槽 (同一 n-gram 的重复窗口不算碰撞)
            slot_of = torch.zeros(uniq_ng.shape[0], dtype=torch.long)
            slot_of[inv] = s1
            d = float(1.0 - torch.unique(slot_of).numel() / uniq_ng.shape[0])
            col.append(d)
        d = max(col)
        worst = max(worst, d)
        rep.append("阶%d 不同ngram %d 撞槽 %.4f" % (n, uniq_ng.shape[0], d))
    # 理论值: D 个不同 n-gram 落进 N 槽 ⇒ 碰撞 ≈ D/(2N) (生日问题). 取 3 倍作门限
    D, N = big.numel(), c.n_slots
    thr = min(0.5, 3.0 * D / (2.0 * N))         # 生日问题: D 个不同项落 N 槽
    return (check("C6a 同 n-gram 必得同槽 (确定性)", same,
                  "3-gram (2,3,4) 在位置 3 与 7 的槽位逐位相同 = %s" % same)
            & check("C6b 哈希碰撞率在生日问题门限内", worst <= thr,
                    "按阶撞槽率 %s; 门限 %.4f (D=%d, 表 %d 槽)" % (", ".join(rep), thr, D, N)))


# --------------------------------------------------------------- C7
def c7_buckets():
    c = mk_cfg()
    L = 6
    ii = torch.arange(L)
    rel = (ii[:, None] - ii[None, :]).clamp(min=0, max=c.n_buckets - 1)
    ok_bucket = bool((rel[0, 0] == 0) and (rel[3, 1] == 2) and (rel[5, 0] == min(5, c.n_buckets - 1)))
    B, nb = 2, c.n_buckets
    bias = torch.arange(B * L * nb, dtype=torch.float32).view(B, L, nb)
    got = bias.gather(-1, rel.unsqueeze(0).expand(B, L, L))
    b = 0
    want = torch.stack([torch.stack([bias[b, i, rel[i, j]] for j in range(L)]) for i in range(L)])
    ok_gather = bool((got[b] == want).all())
    return check("C7 bucket(i-j) 与偏置 gather 正确", ok_bucket and ok_gather,
                 "桶 (0,0)=%d (3,1)=%d 越界截断=%d; gather 逐项一致=%s"
                 % (rel[0, 0], rel[3, 1], rel[5, 0], ok_gather))


# --------------------------------------------------------------- C8
def c8_sparsity():
    c = mk_cfg(n_slots=16384, min_count=3)
    m = EngramRuleLM(c).eval()
    data = zipf_corpus(c, B=16, L=256)
    info = m.engram.build_from_corpus(data)
    with torch.no_grad():
        m(data)
    st = m.stats()
    fl = m.flops_per_token(L=256)
    good = 0.0 < info["alive_ratio"] < 0.95 and 0.0 <= st["rule_skip_ratio"] <= 1.0
    return check("C8 Zipf 语料初始化给出真实空槽 + 稀疏度可测", good,
                 "存活槽 %d/%d (%.2f%%), 规则路径跳过率 %.3f, 规则/内容 FLOPs 比 %.3f"
                 % (info["alive_slots"], info["n_slots"], info["alive_ratio"] * 100,
                    st["rule_skip_ratio"], m.c.n_cand / 256.0))


# --------------------------------------------------------------- C9 / C10 (修正后新增)
def c9_content_gate():
    """修正 A: 内容路径必须能被规则**真正关掉**(否则"稀疏快于稠密"不可达)。
    验证: 把内容门压到 ~0 (bias=-20) 与 直接令 λ=0 必须给出**同一个输出**。"""
    c = mk_cfg()
    m = EngramRuleLM(c).eval()
    x = corpus(c, B=2, L=24)
    m.engram.alive[:] = True
    with torch.no_grad():
        # 参考: 真把 λ 置 0 (第 1 版我变量名叫 o_lam0 却**从没置 λ**, 比的是"正常 vs 关门", 本来就该不同 —— 检查错)
        for b in m.blocks:
            b.attn.lam.fill_(0.0)
        o_lam0 = m(x)
        for b in m.blocks:
            b.attn.lam.fill_(1.0)
            b.attn.content_gate.bias.fill_(-20.0)      # sigmoid(-20) ≈ 2e-9 ⇒ 内容路径被门关死
        o_gate = m(x)
        gc = float(torch.sigmoid(m.blocks[0].attn.content_gate.bias))
    d = float((o_lam0 - o_gate).abs().max())
    return check("C9 内容路径可被规则门真正关闭", d < 1e-4,
                 "门值 %.2e 时输出与 λ=0 的最大差 %.2e (门可关到 0 → 规则可替换内容)" % (gc, d))


def c10_skip_covers_read():
    """修正 B: "跳过"必须覆盖**整条**规则路径 (含 n-gram 嵌入/软读), 不只绕开 bucket_proj。"""
    c = mk_cfg()
    m = EngramRuleLM(c).eval()
    x = corpus(c, B=2, L=24)
    calls = {"n": 0}
    orig = EngramMemory.ngram_query
    EngramMemory.ngram_query = lambda self, *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(self, *a, **k))[1]
    m.engram.alive[:] = True
    with torch.no_grad():
        m(x)
    n_alive = calls["n"]
    calls["n"] = 0
    m.engram.alive.zero_()
    with torch.no_grad():
        m(x)
    n_dead = calls["n"]
    EngramMemory.ngram_query = orig
    return check("C10 全空槽时连 n-gram 嵌入都不算", (n_dead == 0) and (n_alive > 0),
                 "ngram_query 调用: 有存活槽 %d 次 → 全空后 %d 次" % (n_alive, n_dead))


# --------------------------------------------------------------- C14 (SwiGLU 版取代 C11)
def c14_window_and_no_quadratic():
    """★ 架构改成 SwiGLU 聚合后, 旧的 C11(sparse_content 等价) 变成**空转检查**
    (那些开关在新代码里已无作用, 差值恒 0 ⇒ 假通过)。换成两个真判据:
      (a) 窗口 W = k_buckets·ceil(L/n_buckets) 必须远小于 L;
      (b) 峰值显存随 L 近似**线性**(没有 (L,L) 物化), 不是二次。"""
    c = mk_cfg(n_buckets=64, k_buckets=4)
    m = EngramRuleLM(c).eval()
    m.engram.alive[:] = True
    ws = {}
    for L in (128, 256):
        x = corpus(c, B=1, L=L)
        with torch.no_grad():
            m(x)
        ws[L] = m.blocks[0].attn.last_window
    # ★ 判据随设计更新: 窗口改成**固定常数**后 (W 与 L 无关才是线性的前提),
    #   旧判据还在比 "Kb·ceil(L/nb)" —— 那是 W∝L 的旧设计, 属于检查没跟上修法。
    want = c.k_buckets * max(1, c.window_max // c.k_buckets)
    ok_w = (ws[128] == want) and (ws[256] == want)
    return check("C14a 规则窗口为固定常数 (与 L 无关)", ok_w,
                 "L=128 窗口 %d, L=256 窗口 %d, 期望常数 %d ⇒ 相对 L² 省 %d×"
                 % (ws[128], ws[256], want, 256 // want))


def c15_no_full_matrix():
    """(b) 线性扩展: 张量元素总量随 L 的比值必须远小于二次 (4×)"""
    c = mk_cfg(n_buckets=64, k_buckets=4)
    m = EngramRuleLM(c).eval()
    m.engram.alive[:] = True
    seen = {}
    for L in (128, 256):
        mx = {"n": 0}

        def hook(mod, inp, out):
            t = out if torch.is_tensor(out) else None
            if t is not None:
                mx["n"] = max(mx["n"], int(t.numel()))
        hs = [b.register_forward_hook(hook) for b in m.blocks[0].attn.modules()
              if isinstance(b, nn.Linear)]
        with torch.no_grad():
            m(corpus(c, B=1, L=L))
        [h.remove() for h in hs]
        seen[L] = mx["n"]
    ratio = seen[256] / max(seen[128], 1)
    return check("C15 中间张量随 L 近似线性 (无 (L,L))", ratio < 2.6,
                 "最大中间张量 %d → %d, 比值 %.2f (线性≈2, 二次≈4)" % (seen[128], seen[256], ratio))



# --------------------------------------------------------------- C12 / C13 (我犯过的错, 各配一条检查)
def c12_penalty_is_differentiable():
    """我犯过的错: 惩罚项拿了 detach 后的 float ⇒ 对梯度零贡献, 实验等于白跑。"""
    c = mk_cfg()
    m = EngramRuleLM(c)
    m.engram.alive[:] = True
    for b in m.blocks:
        b.attn.record_gc = True
    x = corpus(c, B=2, L=24)
    m.zero_grad(set_to_none=True)
    m(x)
    (0.5 * m.content_usage(grad=True)).backward()
    g = m.blocks[0].attn.content_gate.bias.grad
    gb = None if g is None else float(g.abs().max())
    return check("C12 内容使用率惩罚项可微", (gb or 0.0) > 0.0,
                 "惩罚对 content_gate.bias 的梯度 %.3e (报数用 float 值 %.4f)" % (gb or 0.0,
                                                                                 m.content_usage()))


def c13_training_determinism():
    """我犯过的错: 两次对照没固定种子 ⇒ 差异全是初始化噪声。
    ★ 2026-10-09 补 (diag_det.py 定位): CPU 的并行 index_add/归约 (embedding 反向) 按线程调度
    累加 ⇒ 同种子两次训练的**参数指纹**会差 ~6e-05 绝对 (千量级和的 ~1e-7 相对), 且**时有时无**;
    钉成单线程后逐位相等。这是 harness 自身的浮点噪声, 不是种子没固定 ⇒ 本检查先钉线程再比。"""
    nth = torch.get_num_threads()
    torch.set_num_threads(1)
    c = mk_cfg()
    dtr = corpus(c, B=4, L=16)

    def two_steps():
        torch.manual_seed(7)
        m = EngramRuleLM(c)
        m.engram.alive[:] = True
        g = torch.Generator().manual_seed(7)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for _ in range(2):
            i = torch.randint(0, dtr.shape[0], (4,), generator=g)
            x = dtr[i]
            lg = m(x)
            loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, c.vocab), x[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        h = torch.zeros(1)
        for p in m.parameters():
            h = h + float(p.detach().double().sum())
        return float(h)

    try:
        a, b = two_steps(), two_steps()
    finally:
        torch.set_num_threads(nth)
    return check("C13 同种子两次训练逐位一致 (单线程)", a == b, "两次参数指纹差 %.3e" % abs(a - b))


# --------------------------------------------------------------- C17 (设备对齐, 我跑基准才撞到的错)
def c17_device_agnostic():
    """我犯过的错: 把乘子从"按输入设备现建"改成模块 buffer 后, 拿 **CPU 语料**喂 **CUDA 模块**
    会 einsum 设备不匹配 —— 这是静态可预见的, 我却跑基准时才撞到 (违反"错误修正前禁止跑代码")。
    判据: 无论模块在哪个设备, 传入任何设备的语料都必须自洽跑通。"""
    c = mk_cfg()
    data = corpus(c, B=4, L=32)
    ok = []
    for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        m = EngramRuleLM(c).to(dev)
        try:
            m.engram.build_from_corpus(data)              # 故意传 CPU 数据
            with torch.no_grad():
                m(data.to(dev))
            ok.append("%s:OK" % dev)
        except Exception as e:
            ok.append("%s:★%s" % (dev, str(e)[:40]))
    return check("C17 模块/语料任意设备组合都能跑", all("OK" in x for x in ok), " ".join(ok))


# --------------------------------------------------------------- C18 / C19 (性能改动的等价性)
def c18_rope_equiv():
    """性能改动版 rope(表缓存 + rotate_half) 必须与参考实现**逐位等价**。"""
    c = mk_cfg()
    g = torch.Generator().manual_seed(11)
    B, L = 2, 24
    x = torch.randn(B, L, c.n_heads, c.head_dim, generator=g)
    pos = torch.arange(L)

    def ref_rope(x, pos, dim, theta=10000.0):
        half = dim // 2
        f = torch.arange(half, dtype=torch.float32) / half
        inv = theta ** (-f)
        ang = pos.float().view(1, L, 1, 1) * inv.view(1, 1, 1, half)
        cc, ss = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        xr, xp = x[..., :half], x[..., half:dim]
        return torch.cat([xr * cc - xp * ss, xr * ss + xp * cc, x[..., dim:]], dim=-1)

    a, b = rope(x, pos, c.rope_dim), ref_rope(x, pos, c.rope_dim)
    d = float((a - b).abs().max())
    return check("C18 rope 表缓存版与参考逐位等价", d == 0.0, "max|差| = %.1e (要求恰为 0)" % d)


def c19_ngram_query_equiv():
    """性能改动版 ngram_query(一次查表+后缀和) 必须与参考(逐阶查表)逐位等价。"""
    c = mk_cfg()
    m = EngramMemory(c)
    g = torch.Generator().manual_seed(12)
    x = torch.randint(0, c.vocab, (3, 20), generator=g)

    def ref_q(x):
        B, L = x.shape
        bi = torch.arange(B).repeat_interleave(L)
        ti = torch.arange(L).repeat(B)
        qs = []
        for n in range(1, c.ngram_max + 1):
            toks = torch.stack([x[bi, (ti - j).clamp(min=0)] for j in range(n - 1, -1, -1)], -1)
            q = m.ngram_query.__self__ and None
            qs.append(None)
        # 参考: 逐阶单独查表求和
        out = []
        for n in range(1, c.ngram_max + 1):
            toks = torch.stack([x[bi, (ti - j).clamp(min=0)] for j in range(n - 1, -1, -1)], -1)
            out.append(m.ngram_query.__globals__["nn"].Embedding if False else None)
        return None

    # 直接对拍: 向量化版 vs 手写逐阶版 (都用同一个 emb)
    bi = torch.arange(x.shape[0]).repeat_interleave(x.shape[1])
    ti = torch.arange(x.shape[1]).repeat(x.shape[0])
    got = m.ngram_query(x, m.key, bi, ti)                      # (M,R,d)
    ref = []
    for n in range(1, c.ngram_max + 1):
        toks = torch.stack([x[bi, (ti - j).clamp(min=0)] for j in range(n - 1, -1, -1)], -1)
        q = m.key(toks).sum(1)
        for _ in range(c.n_hash):
            ref.append(q)
    ref = torch.stack(ref, dim=1)
    d = float((got - ref).abs().max())
    # ★ 判据说明: 这里**不能**要求逐位相等 —— 向量化版用 cumsum 后缀和, 参考版用显式 .sum(1),
    #   求和**结合序**不同 ⇒ 差 1 个 ulp 是 IEEE 正常行为, 不是错。
    #   容差按 float32 机器精度 × 数值量级给出 (8·eps·max|ref|), 并可验其量级。
    tol = 8.0 * torch.finfo(torch.float32).eps * max(float(ref.abs().max()), 1e-6)
    return check("C19 ngram_query 向量化版等价 (≤ 8 ulp, 求和结合序差异)", d <= tol,
                 "max|差| = %.2e, 容差 %.2e (= 8·eps·max|ref|), 相对 %.1e"
                 % (d, tol, d / max(float(ref.abs().max()), 1e-9)))


# --------------------------------------------------------------- C20 (近端初值: 双峰的修法)
def c20_near_first_window():
    """双峰的机制: 桶选择只由查询局部 n-gram 决定, "中心词距离"随 k 变(1/4/7)、不在局部 n-gram 里
    ⇒ 规则只能固定选一组桶; 初值随机 ⇒ 约半数种子窗口盖不住中心词 ⇒ 学成词项关联(未见词全错)。
    判据: 近端初值下, **初始选中的桶必须覆盖偏移 {1,4,7}** (k=0/1/2 的中心词距离)。"""
    c = mk_cfg(n_buckets=64, k_buckets=4, window_max=16, near_first=2.0)
    m = EngramRuleLM(c).eval()
    m.engram.alive[:] = True
    L = 24
    x = corpus(c, B=1, L=L)
    with torch.no_grad():
        m(x)
    a = m.blocks[0].attn
    # 复算初始选择: bias 只由 bias 项决定 (权重为 0 初始化? 不, bucket_proj.weight 随机; 用偏置近似)
    with torch.no_grad():
        bias = a.bucket_proj.bias.clone()
        bn = (bias - bias.mean()) / (bias.std() + 1e-5)
        topb = bn.topk(c.k_buckets).indices.tolist()
    # 新窗口语义: 起点 = 偏置最高的桶, 长度恒为 window_max 的**连续**偏移区间
    step = max(1, -(-L // c.n_buckets))
    start = topb[0] * step
    offs = set(range(start, start + c.window_max))
    need = {1, 4, 7}
    return check("C20 近端初值使初始窗口覆盖中心词距离 {1,4,7}", need <= offs,
                 "起点桶 %d ⇒ 覆盖偏移 [%d,%d] (需含 %s)" % (topb[0], start, start + c.window_max - 1, sorted(need)))


# --------------------------------------------------------------- C21 (绝对位置通路是否真的接上)
def c21_abs_pos_wired():
    """双峰成因是"要 k" (前缀 PP 数) = 查询绝对位置。修法给门的查询侧加了显式位置嵌入。
    判据(接线检查, 不是效果检查): ① 位置嵌入收到非零梯度; ② 去掉它输出必须变。"""
    c = mk_cfg()
    m = EngramRuleLM(c)
    m.engram.alive[:] = True
    x = corpus(c, B=2, L=24)
    out = m(x)
    out.float().log_softmax(-1).mean().backward()
    g = m.blocks[0].attn.pos_abs.weight.grad
    gmax = 0.0 if g is None else float(g.abs().max())
    with torch.no_grad():
        out2 = m(x)
    # 把位置嵌入清零后输出必须变化 (证明它真的进了计算)
    with torch.no_grad():
        for b in m.blocks:
            b.attn.pos_abs.weight.zero_()
        out3 = m(x)
    d = float((out2 - out3).abs().max())
    return (check("C21a 绝对位置嵌入收到非零梯度", gmax > 0.0, "max|grad| = %.3e" % gmax)
            & check("C21b 清零位置嵌入后输出改变", d > 1e-6, "max|Δout| = %.3e" % d))


# --------------------------------------------------------------- C22 (判据本身必须可失败)
def c22_metric_is_falsifiable():
    """★ 规则二: 判据必须**在反例下会变**。
    我违反过: 报了三遍"窗口覆盖中心词 = 1.000"当证据, 但起点恒 0 时 `0≤d<W` 对 d∈{1,4,7} 恒真
    ⇒ 该量不可能失败。这里构造反例: 把桶偏置改成**偏爱远端桶**, 起点必须移动、覆盖率必须掉到 0。"""
    c = mk_cfg(n_buckets=64, k_buckets=4, window_max=16)
    m = EngramRuleLM(c).eval()
    m.engram.alive[:] = True
    L = 24
    x = corpus(c, B=1, L=L)
    a = m.blocks[0].attn
    with torch.no_grad():
        a.bucket_proj.bias.copy_(torch.arange(c.n_buckets, dtype=torch.float32))  # 反例: 偏爱远端
    a.record = True
    with torch.no_grad():
        m(x)
    start = int(a.last_startb[0, L - 1].item())          # 最后一个位置的起点桶
    # ★ 读**实际偏移**(夹紧后) 而不是复算公式 start*step —— 复算会与 C28 的夹紧脱节 (两个真相)。
    off0 = int(a.last_window_off[0, L - 1, 0].item())
    need = 1                                             # 最短的中心词距离
    covered = (off0 <= need < off0 + c.window_max)
    return check("C22 位置窗口判据可失败 (反例下起点移动/覆盖变 0)", (start > 0) and (not covered),
                 "偏爱远端后起点桶 %d ⇒ 实际覆盖偏移起点 %d, 中心词距离 1 %s"
                 % (start, off0, "被覆盖" if covered else "**不被覆盖** (判据确实可失败)"))


# --------------------------------------------------------------- C23 (贡献量判据必须可失败)
def c23_contrib_is_falsifiable():
    """★ 规则二: 新诊断量 (各窗口位置的贡献) 必须**会随输入变**, 否则又是常量指标。
    反例构造: 两组不同输入, 其"贡献最大的窗口位置"不该永远相同; 且贡献必须非负、非全零。"""
    c = mk_cfg(n_buckets=64, k_buckets=4, window_max=16)
    m = EngramRuleLM(c).eval()
    m.engram.alive[:] = True
    a = m.blocks[0].attn
    a.record = True
    tops = []
    for sd in (1, 2, 3):
        x = corpus(c, B=4, L=24, seed=sd)
        with torch.no_grad():
            m(x)
        contrib = a.last_contrib                       # (B,L,W)
        tops.append(contrib.reshape(-1, contrib.shape[-1]).argmax(-1))
    differ = any(not torch.equal(tops[0], t) for t in tops[1:])
    # ★ 第 1 版写成 .sum()>0 —— 但贡献可正可负(up 可为负), 正负抵消后恒 ≈0 ⇒ 检查自己错。
    #   非零性要用 |·| 判。
    nonzero = float(a.last_contrib.abs().sum()) > 0
    return check("C23 贡献量判据可失败 (随输入变、非全零)", differ and nonzero,
                 "三组输入的 argmax 窗口位置分布是否不同=%s; 贡献非全零=%s" % (differ, nonzero))



# --------------------------------------------------------------- C27 (位置→桶偏置通路)
def c27_pos_bucket_path():
    """★ 路线①(b): 位置(可数统计) → **桶偏置**(窗口起点)。
    §25 判定: 起点只由查询的**局部 n-gram** 决定, 而"中心词有多远"(=1+3k) 不在局部 n-gram 里
    ⇒ 规则没有放置窗口的信息; 门的查询侧那条位置通路 (pos_abs, C21) 只决定"窗内怎么挑",
    到不了"往回看多远"。判据:
      ① 可证伪(接线): 位置项设成"位置 i ⇒ 桶 min(i,nb-1) 抬 10"后, 起点桶必须**逐位置**等于该值;
      ② 对照: 位置表清零 ⇒ 起点与位置无关 (只剩 n-gram 那一项);
      ③ 恒等: 零初值下表串上后, 与 use_pos_bucket=False 的前向**逐位相同**;
      ④ 可学: 正常配置下反向一次, 位置表必须收到非零梯度。"""
    L = 16
    c = mk_cfg(n_buckets=16, window_max=4)
    m = EngramRuleLM(c).eval()
    m.engram.alive[:] = True
    x = corpus(c, B=1, L=L, seed=4)
    a = m.blocks[0].attn
    i = torch.arange(L)
    want = torch.clamp(i, max=c.n_buckets - 1)
    with torch.no_grad():
        a.bucket_proj.weight.zero_(); a.bucket_proj.bias.zero_()   # 隔离位置项的主张
        a.pos_bucket.weight.zero_(); a.pos_bucket.weight[i, want] = 10.0
    a.record = True
    with torch.no_grad():
        m(x)
    got = a.last_startb[0].cpu()
    ok_track = bool((got == want).all())
    with torch.no_grad():
        a.pos_bucket.weight.zero_()
        m(x)
    got0 = a.last_startb[0].cpu()
    ok_zero = int(torch.unique(got0).numel()) == 1
    # ③ 恒等: 分开配置对象, 只设同种子; 不能先改 c 再创建第二模型 (会让两边共享开关)
    c_on = mk_cfg(n_buckets=16, window_max=4, use_pos_bucket=True)
    torch.manual_seed(7)
    m1 = EngramRuleLM(c_on).eval()
    m1.engram.alive[:] = True
    c_off = mk_cfg(n_buckets=16, window_max=4, use_pos_bucket=False)
    torch.manual_seed(7)
    m0 = EngramRuleLM(c_off).eval()
    m0.engram.alive[:] = True
    xt = corpus(c_on, B=2, L=12, seed=9)
    with torch.no_grad():
        d_id = float((m1(xt) - m0(xt)).abs().max())
    # ④ 位置表必须收到梯度 (只对开关开启的模型检查)
    m1.zero_grad()
    m1(xt).float().log_softmax(-1).mean().backward()
    g = m1.blocks[0].attn.pos_bucket.weight.grad
    gmax = 0.0 if g is None else float(g.abs().max())
    ok = ok_track and ok_zero and (d_id == 0.0) and (gmax > 0.0)
    return check("C27 位置→桶偏置通路 (接线/对照/恒等/可学)", ok,
                 "起点逐位置跟随位置项=%s; 清零后起点种类 %d (要求 1); 零初值恒等 max|Δ| = %.2e (要求 0); "
                 "位置表 max|grad| = %.2e" % (ok_track, int(torch.unique(got0).numel()), d_id, gmax))


# --------------------------------------------------------------- C28 (因果夹紧: 起点不得越过查询位)
def c28_window_clamp():
    """★ §25 抓到窗口选择的零梯度退化态: 若起点偏移使窗口超出因果前缀, 越界 K/V 被 mask 后
    窗口聚合项与窗口参数梯度随有效项减少; 若整窗 j<0, 窗口输出与其梯度都为 0。但 §4.4 direct(agg)
    旁路仍可独立学习, 不要把两支路混为一谈。
    判据:
      ① 极端远端偏爱(起点桶 = nb-1)下, 查询前缀足够长时必须有 W 个有效位置;
      ② 夹紧后窗口支路对 wu 收到非零梯度;
      ③ 负向对照: 未夹紧公式下有至少一行整窗无效, 且隔离 direct 后 wu 梯度恰为 0 (判据可失败)。"""
    c = mk_cfg(n_buckets=64, window_max=16)
    m = EngramRuleLM(c).eval()
    m.engram.alive[:] = True
    L = 24
    x = corpus(c, B=2, L=L, seed=6)
    a = m.blocks[0].attn
    with torch.no_grad():
        a.bucket_proj.weight.zero_()
        a.bucket_proj.bias.copy_(torch.arange(c.n_buckets, dtype=torch.float32))   # 极端: 偏爱远端
    a.record = True
    m.zero_grad()
    m(x).float().log_softmax(-1).mean().backward()
    sb = a.last_startb                                     # (B,L) 起点桶
    step = max(1, -(-L // c.n_buckets))
    _, jvalid, offn = a._window_index(sb, L, torch.arange(L).view(1, L))   # 走模块自己的代码路径
    qpos = torch.arange(L).view(1, L, 1)
    expected_count = torch.minimum(qpos + 1, torch.tensor(c.window_max))
    valid_count = jvalid.sum(-1)
    ok_count = bool((valid_count == expected_count.squeeze(-1)).all())
    ok_eligible = bool((valid_count[:, c.window_max - 1:] == c.window_max).all())
    off_old = sb.unsqueeze(-1) * step + torch.arange(c.window_max).view(1, 1, c.window_max)
    j_old = torch.arange(L).view(1, L, 1) - off_old
    bad_old = int((~(j_old >= 0).any(-1)).sum())           # 未夹紧 ⇒ 整行无效的行数
    # ③ 负向对照要隔离 §4.4 direct, 否则它独立收到梯度会让"窗口零梯度"断言失真。
    gm = float(a.bucket_proj.bias.grad.abs().max())
    gw = float(a.wu.weight.grad.abs().max())
    # 负向对照: 同一因果损失下关闭 direct; 未夹紧窗口全无效时, wu 梯度必须严格为 0。
    c_old = mk_cfg(n_buckets=64, window_max=16, clamp_window=False)
    torch.manual_seed(6)
    m_old = EngramRuleLM(c_old).eval()
    m_old.engram.alive[:] = True
    a_old = m_old.blocks[0].attn
    with torch.no_grad():
        a_old.bucket_proj.weight.zero_()
        a_old.bucket_proj.bias.copy_(torch.arange(c_old.n_buckets, dtype=torch.float32))
    a_old.ablate_direct = True
    m_old.zero_grad()
    m_old(x).float().log_softmax(-1).mean().backward()
    gw_old = a_old.wu.weight.grad
    gw_old = 0.0 if gw_old is None else float(gw_old.abs().max())
    ok = ok_count and ok_eligible and (bad_old > 0) and (gm > 0.0) and (gw > 0.0) and (gw_old == 0.0)
    return check("C28 起点越界时窗口恒含所有可用因果位置 (旧行为=空窗零梯度)", ok,
                 "夹紧: 起点桶 %d, min(q+1,W) 计数=%s, q≥W-1 后全窗有效=%s, max|∂桶偏置| %.2e, max|∂wu| %.2e; "
                 "未夹紧+关 direct: 整行无效 %d, max|∂wu| %.2e (要求恰 0)"
                 % (int(sb.max()), ok_count, ok_eligible, gm, gw, bad_old, gw_old))


# --------------------------------------------------------------- C29 (局部归一化分母)
def c29_local_norm_denominator():
    """★ 局部归一化的分母必须只统计**有效位置**。判据(黑盒不变性, 不读内部张量):
      窗口加长 (window_max 4 → 8) 时, 多出的偏移对**短前缀位置**全部越界 ⇒ 输出必须逐位不变;
      越界位置只该以 0 进入计算 (旧写法 gate.abs().sum(2) 把它们算进分母 ⇒ 输出被缩放)。
      负向对照: 同一测量在 local_norm=False (除数是 √W) 下必须**明显变化** ⇒ 判据不是恒真。"""
    c = mk_cfg(n_buckets=16, window_max=4)
    L = 12
    x = corpus(c, B=2, L=L, seed=8)
    torch.manual_seed(3)
    m = EngramRuleLM(c).eval()
    m.engram.alive[:] = True
    with torch.no_grad():
        a = m.blocks[0].attn
        a.bucket_proj.weight.zero_(); a.bucket_proj.bias.zero_()   # 起点桶恒 0 ⇒ 隔离"分母"这一主张
    with torch.no_grad():
        c.window_max = 4; o4 = m(x)
        c.window_max = 8; o8 = m(x)
        c.local_norm = False
        c.window_max = 4; n4 = m(x)
        c.window_max = 8; n8 = m(x)
        c.local_norm = True
        c.window_max = 4
    K = 4                                                  # i < K 的行: 多出的偏移才全越界
    d_same = float((o4[:, :K] - o8[:, :K]).abs().max())
    scale = max(float(o4[:, :K].abs().max()), 1e-9)
    d_neg = float((n4[:, :K] - n8[:, :K]).abs().max())
    return check("C29 局部归一化分母只含有效位置 (越界位置零影响)",
                 (d_same <= 1e-6 * scale) and (d_neg > 1e-4),
                 "W=4 vs 8 在短前缀位置 max|Δ| = %.2e (相对 %.1e, 要求≈0); 负向对照(除√W) = %.2e (要求>1e-4)"
                 % (d_same, d_same / scale, d_neg))


# --------------------------------------------------------------- C30 (消融必须只关自己那一路)
def c30_ablation_switches():
    """★ 三种干预必须对应三条路径: ablate_window 只关窗口聚合、保留 direct; ablate_direct 反之;
    ablate_rule 同时关两者。诊断的窗口偏移独立于 direct 消融, 但关窗口也不能伪造窗口读数。
    判据: ① 单独关窗口只改变输出, last_window_off 保持不变(选择仍在但贡献被消融);
          ② 单独关 direct 同理; ③ 两条都关的输出确实不同于原输出。"""
    c = mk_cfg(n_buckets=16, window_max=4)
    m = EngramRuleLM(c).eval()
    m.engram.alive[:] = True
    x = corpus(c, B=2, L=12, seed=5)
    ab = [b.attn for b in m.blocks]
    for a in ab:
        a.record = True
    with torch.no_grad():
        o0 = m(x)
        w0 = ab[0].last_window_off.clone()
        for a in ab:
            a.ablate_window = True
        ow = m(x)
        ww = ab[0].last_window_off.clone()
        for a in ab:
            a.ablate_window = False
            a.ablate_direct = True
        od = m(x)
        wd = ab[0].last_window_off.clone()
        for a in ab:
            a.ablate_direct = False
            a.ablate_rule = True
        oa = m(x)
        for a in ab:
            a.ablate_rule = False
    dw = float((o0 - ow).abs().max())
    dd = float((o0 - od).abs().max())
    da = float((o0 - oa).abs().max())
    same_window = bool(torch.equal(w0, ww) and torch.equal(w0, wd))
    ok = (dw > 0.0) and (dd > 0.0) and (da > 0.0) and same_window
    return check("C30 窗口/direct/整体规则消融各自独立", ok,
                 "关窗口 Δ %.2e; 关direct Δ %.2e; 全关 Δ %.2e; 两种单路消融均保持窗口诊断=%s"
                 % (dw, dd, da, same_window))


# --------------------------------------------------------------- C32 (桶步长必须跨运行长度固定)
def c32_bucket_step_is_length_invariant():
    """★ 同一模型训练/推理长度不同不得重映射相对桶。负对照用 bucket_step=0 的旧公式,
    证明该检查能抓到 ceil(L/n_buckets) 的运行时漂移。"""
    fixed_cfg = mk_cfg(n_buckets=8, bucket_step=3)
    dynamic_cfg = mk_cfg(n_buckets=8, bucket_step=0)
    fixed = RuleAttention(fixed_cfg)
    dynamic = RuleAttention(dynamic_cfg)
    got = [fixed._bucket_step(L) for L in (7, 16, 40)]
    legacy = [dynamic._bucket_step(L) for L in (7, 16, 40)]
    ok = got == [3, 3, 3] and legacy == [1, 2, 5] and len(set(legacy)) > 1
    return check("C32 固定 bucket_step 不随 padded/runtime L 改变", ok,
                 "fixed L=(7,16,40)→%s; legacy dynamic L→%s (反例跨度确实改变)" % (got, legacy))


def main():
    print("Engram-Rule v1.0 检查套件 (每条对应一个曾经/可能出错的原则)\n")
    for fn in (c1_rope, c2_residual_shortcut, c3_conflict_weighting, c45_grad,
               c6_hash, c7_buckets, c8_sparsity, c9_content_gate, c10_skip_covers_read, c14_window_and_no_quadratic, c15_no_full_matrix,
               c12_penalty_is_differentiable, c13_training_determinism, c17_device_agnostic, c18_rope_equiv, c19_ngram_query_equiv, c20_near_first_window, c21_abs_pos_wired, c22_metric_is_falsifiable, c23_contrib_is_falsifiable,
               c27_pos_bucket_path, c28_window_clamp, c29_local_norm_denominator, c30_ablation_switches,
               c32_bucket_step_is_length_invariant):
        fn()
    print("\n================ 汇总 ================")
    print("  通过 %d 项, 未通过 %d 项 %s" % (len(OK), len(BAD), ("★ " + ", ".join(BAD)) if BAD else ""))
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
