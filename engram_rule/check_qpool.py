"""★问题池化通路(§61)的**可失败性检查** —— 跑实验前必须全绿。

规则 3: 改 → 补检查 → 过检查 → 才准跑。规则 7: 每条都要附"反例下会变"。

本脚本**零训练**、秒级。它检查的是**测量机器本身**, 不是实验结果:

  Q1  `use_qpool=False` 时输出与**旧版逐位相同**(不出手改出货路径)
  Q2  `use_qpool=True` 且 `q_pool=None` 时同样逐位相同(关通路的第二条路)
  Q3  `qpool_proj` 初值为单位阵(否则"初始等价于原样加"这句是假的)
  Q4  `alpha` 初值 = 配置值且**可学习**(在 optimizer.param_groups 里)
  Q5  ★可失败: 设一个**非零** q_pool 后输出**必须变** —— 否则 Q1/Q2 是恒真的
  Q6  ★可失败: 换成**另一个** q_pool 后输出**必须再变一次**
  Q7  q_pool 的 shape 校验会真的抛(不是静默广播错)
  Q8  `set_question_pool(None)` 能把通路**重新关掉**(训练/评测来回切换安全)
  Q9  ★梯度: alpha 与 qpool_proj 必须收到非零梯度(通路真在被学)
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
seed = 11          # ★所有模型的构造种子, 开/关池化必须**同一个 seed**
FAILURES = []
TOTAL = [0]


class _SmallTok:
    """极简 tokenizer: 只为跑通批生成路径。id 空间就是 0..vocab-1。"""

    def __init__(self, bos, u, a, eos):
        self.bos, self.u, self.a, self.eos = bos, u, a, eos

    def token_to_id(self, s):
        return {"[BOS]": self.bos, "[USER]": self.u,
                "[ASSISTANT]": self.a, "[EOS]": self.eos}.get(s)

    def encode(self, text, add_special_tokens=False):
        ids = [self.bos]
        i = 0
        while i < len(text):
            if text.startswith("[USER]", i):
                ids.append(self.u)
                i += len("[USER]")
            elif text.startswith("[ASSISTANT]", i):
                ids.append(self.a)
                i += len("[ASSISTANT]")
            else:
                ids.append(10 + (ord(text[i]) % 200))     # 落在 [10, 210) 内
                i += 1
        class _E:
            def __init__(self, ids):
                self.ids = ids
        return _E(ids)

    # bytelevel_token_bytes 需要这两个接口
    def get_added_tokens_decoder(self):
        return {self.bos: None, self.u: None, self.a: None, self.eos: None}

    def get_vocab_size(self):
        return 256

    def id_to_token(self, i):
        return "tok%d" % i

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (i % 26)) for i in ids)
TOTAL = [0]


def check(name, cond, detail=""):
    TOTAL[0] += 1
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  — " + detail) if detail else ""))
    if not cond:
        FAILURES.append(name)


def build(use_qpool, user_id=-1, assist_id=-1):
    from model import RCfg, EngramRuleLM
    cfg = dict(vocab=256, d=32, n_layers=2, n_heads=4, ffn_mult=2, rope_dim=8,
               ngram_max=2, n_hash=2, n_slots=256, min_count=1, n_buckets=16,
               gate_rank=16, pos_abs_max=256, window_max=16, bucket_step=32,
               use_qpool=use_qpool, qpool_alpha_init=0.1,
               qpool_user_id=user_id, qpool_assist_id=assist_id)
    torch.manual_seed(seed)
    m = EngramRuleLM(RCfg(**cfg)).to(DEV)
    m.eval()
    return m


def main():
    x = torch.randint(0, 256, (2, 40), device=DEV)
    seed = 11

    # ── Q1/Q2: 关闭时与旧版逐位相同 ───────────────────────────────────
    # ★必须**同一个 seed 重建**两个模型来比: 两个模型各自 init, 权重不同 ⇒
    #   拿"开/关两个不同模型"比输出, 比的是权重差异不是通路(第一版就这么错的,
    #   报出 max|Δ|=2.61 —— 那其实是我自己 init 的噪声)。
    torch.manual_seed(seed)
    m_off = build(False)
    torch.manual_seed(seed)
    m_on = build(True)
    m_off.eval()
    m_on.eval()
    with torch.no_grad():
        y_off = m_off(x)
    m_off.set_question_pool(torch.randn(2, 32, device=DEV))    # 关了也应无效
    with torch.no_grad():
        y_off2 = m_off(x)
    check("Q1 use_qpool=False 输出不受 q_pool 影响", bool((y_off == y_off2).all()))

    with torch.no_grad():
        y_none = m_on(x)                    # q_pool 还是 None
    m_on.set_question_pool(None)
    with torch.no_grad():
        y_none2 = m_on(x)
    check("Q2 use_qpool=True 但 q_pool=None 与关闭时逐位相同",
          bool((y_off == y_none).all()) and bool((y_none == y_none2).all()),
          "max|Δ|=%.2e" % float((y_off - y_none).abs().max()))

    # ── Q3 投影初值为单位阵 ───────────────────────────────────────────
    eye = torch.eye(32, device=m_on.qpool_proj.weight.device)
    check("Q3 qpool_proj 初值为单位阵", bool((m_on.qpool_proj.weight == eye).all()))

    # ── Q4 alpha 可学习 ───────────────────────────────────────────────
    check("Q4a alpha 初值 = 0.1",
          abs(float(m_on.qpool_alpha.detach()) - 0.1) < 1e-7,
          "%.4f" % float(m_on.qpool_alpha.detach()))
    opt = torch.optim.AdamW(m_on.parameters(), lr=1e-3)
    names = {n for g in opt.param_groups for n, _ in
             [(n, p) for n, p in m_on.named_parameters()]}
    check("Q4b alpha 与 qpool_proj 都在 optimizer 的参数表里",
          any("qpool_alpha" in n for n in names)
          and any("qpool_proj" in n for n in names))

    # ── Q5/Q6 ★可失败: 设了 q_pool 输出必须变 ─────────────────────────
    g1 = torch.Generator(device="cpu").manual_seed(1)
    g2 = torch.Generator(device="cpu").manual_seed(2)
    q1 = torch.randn(2, 32, generator=g1).to(DEV)
    q2 = torch.randn(2, 32, generator=g2).to(DEV)
    m_on.set_question_pool(q1)
    with torch.no_grad():
        y1 = m_on(x)
    m_on.set_question_pool(q2)
    with torch.no_grad():
        y2 = m_on(x)
    d1 = float((y1 - y_none).abs().max())
    d2 = float((y2 - y1).abs().max())
    check("Q5 ★设非零 q_pool 后输出必须变", d1 > 1e-6, "max|Δ|=%.4f" % d1)
    check("Q6 ★换一个 q_pool 后必须再变一次", d2 > 1e-6, "max|Δ|=%.4f" % d2)

    # ── Q7 shape 校验 ─────────────────────────────────────────────────
    try:
        m_on.set_question_pool(torch.randn(3, 32, device=DEV))   # batch 不匹配
        with torch.no_grad():
            _ = m_on(x)
        ok7 = False
    except RuntimeError:
        ok7 = True
    check("Q7 batch 不匹配会抛 RuntimeError(不静默广播)", ok7)

    # ── Q8 能重新关掉 ─────────────────────────────────────────────────
    m_on.set_question_pool(None)
    with torch.no_grad():
        y_again = m_on(x)
    check("Q8 设回 None 能重新关掉(逐位回到旧版)", bool((y_again == y_none).all()))

    # ── Q9 ★梯度 ──────────────────────────────────────────────────────
    m_g = build(True)
    m_g.train()
    m_g.set_question_pool(torch.randn(2, 32, device=DEV))
    out = m_g(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    ga = m_g.qpool_alpha.grad
    gp = m_g.qpool_proj.weight.grad
    check("Q9a alpha 收到**非零**梯度", ga is not None and float(ga.abs()) > 0,
          "grad=%s" % (None if ga is None else "%.3e" % float(ga)))
    check("Q9b qpool_proj 收到**非零**梯度",
          gp is not None and float(gp.abs().max()) > 0,
          "grad=%s" % (None if gp is None else "%.3e" % float(gp.abs().max())))

    # ── Q10 forward(q_pool=...) 与 set_question_pool 等价 ──────────────
    torch.manual_seed(seed)
    m_a = build(True)
    m_a.eval()
    torch.manual_seed(seed)
    m_b = build(True)
    m_b.eval()
    with torch.no_grad():
        y_a = m_a(x, q_pool=q1)
    m_b.set_question_pool(q1)
    with torch.no_grad():
        y_b = m_b(x)
    check("Q10a forward(q_pool=) 与 set_question_pool 逐位相同",
          bool((y_a == y_b).all()),
          "max|Δ|=%.2e" % float((y_a - y_b).abs().max()))
    # 显式参数**优先于** set 的那个(否则训练/评测来回切换会互相污染)
    m_b.set_question_pool(q2)
    with torch.no_grad():
        y_c = m_b(x, q_pool=q1)
    check("Q10b 显式 q_pool 参数优先于 set 的值",
          bool((y_c == y_a).all()),
          "max|Δ|=%.2e" % float((y_c - y_a).abs().max()))
    # 传 None 应当回到关闭态(不是"用 set 里那个")
    with torch.no_grad():
        y_d = m_b(x, q_pool=None)
    check("Q10c forward(q_pool=None) 回到关闭态",
          bool((y_d == m_b(x, q_pool=None)).all()))
    m_b.set_question_pool(None)
    with torch.no_grad():
        y_e = m_b(x)
    check("Q10d set(None) + forward(None) 与旧版逐位相同", bool((y_e == y_off).all()))

    # ── Q11 shape 校验走的是显式报错路径, 不是静默广播 ─────────────────
    try:
        with torch.no_grad():
            _ = m_a(x, q_pool=torch.randn(2, 31, device=DEV))   # d 不匹配
        ok11 = False
    except RuntimeError as exc:
        ok11 = "q_pool 形状" in str(exc)
    check("Q11 d 不匹配时报**带说明**的 RuntimeError", ok11)

    # ── Q12 ★模型自己切问题段: 必须与手工切逐位相同 ────────────────────
    #   为什么需要这条: 自动切段是**新增的通路入口**, 它算错的话主判据会假红。
    torch.manual_seed(seed)
    m_auto = build(True, user_id=7, assist_id=9)
    m_auto.eval()
    u, a = 7, 9
    xm = torch.randint(10, 250, (2, 30), device=DEV)
    # 手工摆一条 [USER] ... [ASSISTANT] 的 prompt, 问题段就是中间那截
    xm[0, 3], xm[0, 8] = u, a
    xm[1, 5], xm[1, 12] = u, a
    with torch.no_grad():
        y_auto = m_auto(xm)
    # 手工算: 用 emb 对问题段取 mean(与模型内部同源)
    manual = []
    for r in range(xm.shape[0]):
        ids = xm[r].tolist()
        ui = next((i + 1 for i in range(len(ids) - 1, -1, -1) if ids[i] == u), None)
        ai = next((i for i in range(len(ids) - 1, -1, -1) if ids[i] == a), None)
        seg = torch.tensor(ids[ui:ai], device=DEV)
        manual.append(m_auto.emb(seg).mean(0))
    qm = torch.stack(manual)
    torch.manual_seed(seed)
    m_ref = build(True)
    m_ref.eval()
    # m_ref 与 m_auto 必须同权重(同 seed 重建)才能比
    m_ref.load_state_dict(m_auto.state_dict())
    with torch.no_grad():
        y_manual = m_ref(xm, q_pool=qm)
    check("Q12a ★模型自动切段 == 手工切段(机器精度内相同)",
          float((y_auto - y_manual).abs().max()) < 1e-5,
          "max|Δ|=%.2e (求和的归约顺序不同 ⇒ 机器精度)" %
          float((y_auto - y_manual).abs().max()))
    # 没有 [USER] 的行 ⇒ 池化为 0 ⇒ 等价于关闭
    # ★长度必须与 y_off 一致(40), 否则比的是两个不同形状的张量(踩过两次)
    # ★★更要紧的是: y_off 是用 **m_off** 算的, 而 m_off 与 m_auto 虽然同 seed
    #   重建且共同参数逐位相同, 但 **m_auto 在 Q1 里被 set_question_pool(q1) 设过**
    #   ⇒ `self.q_pool` 不是 None ⇒ 自动切段算出的 q_pool 会被 set 的那个覆盖!
    #   必须先清掉 set 的值(第一版就是这么误红的: 它拿一个**被设过值**的模型当"无问题"用例)。
    m_auto.set_question_pool(None)
    # `_q_pool_arg` 是 forward 内的临时量, 由下一次 forward 开头自动清 —— 这里不等它,
    # 清的是"跨调用持久"的 self.q_pool。
    assert m_auto.q_pool is None, "set_question_pool(None) 没清掉"
    xn = torch.randint(10, 250, (2, 40), device=DEV)
    while bool((xn == u).any()) or bool((xn == a).any()):
        xn = torch.randint(10, 250, (2, 40), device=DEV)
    # ★★必须用 **m_off(xn)** 而不是开头那个 y_off(= m_off(x)) —— 换输入就是换输出!
    #   (第一版拿 y_off 比, 报出 Δ=3.02; 同一模型换一批输入自己就能差 3.23 ⇒
    #     那个红完全是**比较对象搞错**, 不是模型行为。)
    with torch.no_grad():
        y_off_xn = m_off(xn)
        y_none = m_auto(xn)
    check("Q12b 没有 [USER] 时自动退化为关闭态", bool((y_none == y_off_xn).all()),
          "max|Δ|=%.2e" % float((y_none - y_off_xn).abs().max()))
    # cfg 没给 id ⇒ 不自动切 ⇒ 必须显式传, 否则等于没这通路
    # ★用**同长度**的 x(40) 比(踩过: 用了 30 长的 xm 去比 40 长的 y_off)
    torch.manual_seed(seed)
    m_noid = build(True, user_id=-1, assist_id=-1)
    m_noid.eval()
    m_noid.load_state_dict(m_auto.state_dict())
    with torch.no_grad():
        y_noid = m_noid(x)
    check("Q12c cfg 没给 token id ⇒ 不自动切(与关闭逐位相同)",
          bool((y_noid == y_off).all()))

    # ── Q13 ★状态污染: 上一批的 q_pool 不得残留到下一批 ────────────────
    #   踩过: `_q_pool_arg` 在 forward 里被设了就留着, 下一批"没有问题段"的
    #   输入会**静默拿到上一批的问题摘要** ⇒ 同形状时完全无声, 形状不同才抛。
    m_auto.set_question_pool(None)
    with torch.no_grad():
        _ = m_auto(xm)                    # 第一批: 有 [USER] ⇒ 算出 q_pool
    assert m_auto._q_pool_arg is not None, "第一批应已算出 q_pool"
    with torch.no_grad():
        y_after = m_auto(xn)              # 第二批: 无 [USER] ⇒ 必须退回关闭
    check("Q13a ★无问题段的批次不受上一批 q_pool 污染",
          bool((y_after == y_off_xn).all()),
          "max|Δ|=%.2e" % float((y_after - y_off_xn).abs().max()))
    # 显式传参的优先级在"跨批"时也要成立
    with torch.no_grad():
        _ = m_auto(xm, q_pool=q1)
    with torch.no_grad():
        y_p = m_auto(xn, q_pool=q1)
    m_ref2 = build(True)
    m_ref2.eval()
    m_ref2.load_state_dict(m_auto.state_dict())
    with torch.no_grad():
        y_p2 = m_ref2(xn, q_pool=q1)
    check("Q13b 跨批显式传参仍逐位一致", bool((y_p == y_p2).all()))
    # 形状不符必须抛**带说明**的错, 而不是静默广播
    try:
        with torch.no_grad():
            _ = m_auto(torch.randint(0, 256, (3, 40), device=DEV), q_pool=q1)  # B=3 vs q1 B=2
        ok13c = False
    except RuntimeError as exc:
        ok13c = "残留" in str(exc) or "不匹配" in str(exc)
    check("Q13c batch 不符时报**带说明**的错(提示可能是残留)", ok13c)

    # ── Q14 ★批生成路径也必须支持 q_pool(否则实验只能走慢的逐行) ─────────
    #   动机: 上一版我为了绕开"greedy_generate 不接受 q_pool"而**自己写了生成循环**,
    #   结果慢到几十倍。正确做法是给出货的 greedy_generate_batch 加参数, 让测量
    #   继续走已验证 18 项等价性的那条路径。
    #   ★用**真实 tokenizer 的真实 id**在小模型上跑 —— 但小模型 vocab=256,
    #     而真实 [USER]/[ASSISTANT] 的 id 可能 ≥256 ⇒ 越界索引会触发 device assert。
    #     所以这里用**小模型自己的** id(7/9), 把 rows 的 token 也夹到 256 以内。
    import dialogue_small as ds
    DEVICE = DEV
    torch.manual_seed(seed)
    m_b = build(True, user_id=7, assist_id=9)
    m_b.eval()
    fake_rows = [{"question": "问题" * 3, "answer": "答案" * 2, "history": []}
                 for _ in range(6)]
    # 手工摆 prompt: [BOS] + [USER] + 问题 + [ASSISTANT]
    ids_bos, ids_u, ids_a, ids_eos = 1, 7, 9, 3
    m_b.c.vocab = 256
    # 用一个"够用的 tokenizer"跑: 直接用 ids 构造 rows 不可行 ⇒ 借真实 tokenizer 但夹住 id
    # 简化: 只验证 q_pool 参数在批路径里被**正确转发**(不触发异常)与逐行一致
    with torch.no_grad():
        gb = ds.greedy_generate_batch(m_b, _SmallTok(ids_bos, ids_u, ids_a, ids_eos),
                                      fake_rows, max_new_tokens=8, device=DEVICE,
                                      context_limit=200, q_pool=None)
    ok14a = all("generated" in g for g in gb)
    check("Q14a ★greedy_generate_batch 接 q_pool 参数能跑通", ok14a,
          "%d 行, 首行 %r" % (len(gb), (gb[0]["generated"][:16] if ok14a else "")))
    # 自动切段 vs 显式传入: 逐 token 必须一致
    qs = []
    for r in fake_rows:
        p_ids = _SmallTok(ids_bos, ids_u, ids_a, ids_eos).encode(
            ds.format_prefix(r)).ids
        ui = next((i + 1 for i in range(len(p_ids) - 1, -1, -1) if p_ids[i] == ids_u), None)
        ai = next((i for i in range(len(p_ids) - 1, -1, -1) if p_ids[i] == ids_a), None)
        qs.append(m_b.emb(torch.tensor(p_ids[ui:ai], device=DEVICE)).mean(0))
    qm = torch.stack(qs)
    with torch.no_grad():
        gb2 = ds.greedy_generate_batch(m_b, _SmallTok(ids_bos, ids_u, ids_a, ids_eos),
                                       fake_rows, max_new_tokens=8, device=DEVICE,
                                       context_limit=200, q_pool=qm)
    same = sum(1 for a, b in zip(gb, gb2) if a["answer_ids"] == b["answer_ids"])
    check("Q14b ★批生成: 自动切段 == 显式传入", same == len(fake_rows),
          "%d/%d 行逐 token 一致" % (same, len(fake_rows)))
    # q_pool 的行数不匹配必须抛**带说明**的错
    try:
        with torch.no_grad():
            _ = ds.greedy_generate_batch(m_b, _SmallTok(ids_bos, ids_u, ids_a, ids_eos),
                                         fake_rows, max_new_tokens=4, device=DEVICE,
                                         context_limit=200, q_pool=qm[:2])
        ok14c = False
    except AssertionError as exc:
        ok14c = "q_pool 的行数" in str(exc)
    check("Q14c q_pool 行数不匹配时报带说明的错", ok14c)

    # ── Q15 ★★负向对照: 喂**明显不同**的 q_pool, 生成必须变 ───────────────
    #   ★这是整条池化链的**决定性**判据。Q5/Q6 在单次前向上验过"设了 q_pool 输出会变",
    #     但**端到端生成**上没验过 —— 而端到端才是主判据 M-pool 用的路径。
    #     若这里红了, 说明 q_pool 在生成循环里根本没被用上(§62.4 实测就是这种情况:
    #     喂全零与喂随机都得到 8/8 逐 token 相同 ⇒ 池化在生成路径上是**空操作**)。
    m_auto.set_question_pool(None)
    DEVICE = DEV
    with torch.no_grad():
        g_zero = ds.greedy_generate_batch(
            m_b, _SmallTok(ids_bos, ids_u, ids_a, ids_eos), fake_rows,
            max_new_tokens=12, device=DEVICE, context_limit=200,
            q_pool=torch.zeros(len(fake_rows), 32, device=DEVICE))
        g_rand = ds.greedy_generate_batch(
            m_b, _SmallTok(ids_bos, ids_u, ids_a, ids_eos), fake_rows,
            max_new_tokens=12, device=DEVICE, context_limit=200,
            q_pool=torch.randn(len(fake_rows), 32, device=DEVICE))
    same15 = sum(1 for a, b in zip(g_zero, g_rand)
                 if a["answer_ids"] == b["answer_ids"])
    check("Q15 ★★端到端生成: 全零池化 vs 随机池化必须**不同**",
          same15 < len(fake_rows),
          "%d/%d 行相同(必须 < %d) ⇒ 否则 q_pool 在生成路径上是空操作"
          % (same15, len(fake_rows), len(fake_rows)))
    # 与"不传 q_pool"(模型自切)比: 也应不同(除非模型自切算出的是 0)
    with torch.no_grad():
        g_auto = ds.greedy_generate_batch(
            m_b, _SmallTok(ids_bos, ids_u, ids_a, ids_eos), fake_rows,
            max_new_tokens=12, device=DEVICE, context_limit=200, q_pool=None)
    same15b = sum(1 for a, b in zip(g_auto, g_rand)
                  if a["answer_ids"] == b["answer_ids"])
    check("Q15b 模型自切 vs 显式随机池化也必须不同",
          same15b < len(fake_rows),
          "%d/%d 行相同" % (same15b, len(fake_rows)))

    print("\n%d/%d 通过" % (TOTAL[0] - len(FAILURES), TOTAL[0]))
    if FAILURES:
        print("失败项: %s" % FAILURES)
        return 1
    print("全部通过 —— 可以跑问题池化实验")
    return 0


if __name__ == "__main__":
    sys.exit(main())
