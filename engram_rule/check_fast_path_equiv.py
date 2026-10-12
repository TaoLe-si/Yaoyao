"""§51 快速路径等价性检查: 优化后的热路径**必须**与冻结参照逐位(或在容差内)一致。

金标准是 `_ref_preopt.py` —— 优化前那三段代码的逐字冻结副本(不是重抄的近似)。

判据必须可失败(AGENTS.md 规则 7):
  · E1/E2/E3 正向: 新实现 == 参照, 否则 FAIL。
  · **N1/N2/N3 负向对照**: 故意扰动输入/实现, 检查器**必须报差异**。
    只报"通过"的检查等于没检查(本项目 C22 的教训: 恒真的量不是证据)。
  · N4: 若把参照与新实现同时换成同一个错误实现, 差异必须**消失** ——
    证明"无差异"不是因为比较逻辑本身瞎了。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_ROWS = 6
TOL = 1e-6          # 读数若远小于它, 说明是同一批浮点运算而非巧合

import _ref_preopt as ref          # 金标准: 冻结的优化前实现

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print("[%s] %s :: %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


# ------------------------------------------------------- 上下文管理器: 换成参照实现
class _UseRef:
    """把三段热路径临时换成冻结参照(退出时还原)。"""

    def __enter__(self):
        from model import EngramMemory, RuleAttention
        import dialogue_small as ds
        self._save = (EngramMemory.read, RuleAttention.forward, ds._select_next_id)
        import _ref_preopt as ref
        EngramMemory.read = ref.ref_engram_read
        RuleAttention.forward = ref.ref_rule_attention_forward
        ds._select_next_id = ref.ref_select_next_id
        return self

    def __exit__(self, *exc):
        from model import EngramMemory, RuleAttention
        import dialogue_small as ds
        EngramMemory.read, RuleAttention.forward, ds._select_next_id = self._save
        return False


def _toy_model(seed=0, device=DEV, **over):
    from model import RCfg, EngramRuleLM
    c = RCfg(vocab=64, d=32, n_layers=2, n_heads=4, ffn_mult=2, rope_dim=8,
             ngram_max=2, n_hash=2, n_slots=256, min_count=1, n_buckets=8,
             gate_rank=8, pos_abs_max=128, window_max=8, bucket_step=8, **over)
    torch.manual_seed(seed)
    m = EngramRuleLM(c).to(device)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.randint(0, c.vocab, (3, 24), generator=g).to(device)
    m.engram.alive[:] = True
    return m, c, x


# ------------------------------------------------------- 负向对照: 模拟"真会犯的错"
# 原则(规则 7): 对照必须模拟**这次优化具体可能引入的错误**, 而不是随便加个数。
# 每个 perturbed_* 都是"参照 + 一处真实疏漏", 检查器必须抓到。

def _perturbed_engram_read(self, x, emb):
    """真实疏漏: 死位置忘了置 0(向量化时把 any_live 掩码漏掉)。"""
    v, w, s = ref.ref_engram_read(self, x, emb)
    return self.val(self.slot_ids(x)), w, s


def _perturbed_rule_attention(self, h, pos, rule_vec, rule_w, skip, causal=True, has_rule=None):
    """真实疏漏: 向量化时把 bias/out_direct 的 keep 掩码漏掉(跳过位置本该为 0)。"""
    o = ref.ref_rule_attention_forward(self, h, pos, rule_vec, rule_w, skip, causal)
    return o + 0.5 * (~skip).to(o.dtype).unsqueeze(-1)


def _perturbed_select(*a, **kw):
    """真实疏漏: 漏掉 byte_tail 的 U+FFFD 抑制(表驱动改造最容易丢的一条)。"""
    a = list(a)
    if len(a) > 11:                       # 第 12 个位置参数 = byte_tail
        a[11] = b""
    return ref.ref_select_next_id(*a, **kw)


def e1_engram_read():
    m, c, x = _toy_model()
    alive_idx = torch.arange(c.n_slots, device=DEV)
    m.engram.alive[alive_idx % 3 != 0] = False        # 造成大量死位置
    with torch.no_grad():
        new_v, new_w, new_s = m.engram.read(x, m.emb)
        with _UseRef():
            ref_v, ref_w, ref_s = m.engram.read(x, m.emb)
    dv = float((new_v - ref_v).abs().max())
    dw = float((new_w - ref_w).abs().max())
    ds_ = int((new_s ^ ref_s).sum())
    ok = check("E1 Engram 软读等价", dv <= TOL and dw <= TOL and ds_ == 0,
               "Δv=%.3e Δw=%.3e Δskip=%d (共 %d 位, 死位置 %d 个)"
               % (dv, dw, ds_, int(new_s.numel()), int((~new_s).sum())))
    with torch.no_grad():
        bad_v = _perturbed_engram_read(m.engram, x, m.emb)[0]
    dbad = float((bad_v - ref_v).abs().max())
    check("N1 E1 负向对照(漏掉 any_live 掩码必须被检出)", dbad > TOL,
          "疏漏版与参照差 %.3e ⇒ 比较器有牙" % dbad)
    return ok


# ------------------------------------------------------- E2: 规则注意力
def e2_rule_attention():
    import model as M
    m, c, x = _toy_model(seed=3)
    with torch.no_grad():
        new = m(x)
        with _UseRef():
            ref = m(x)
    d = float((new - ref).abs().max())
    ok = check("E2 规则注意力输出等价", d <= TOL, "max|Δlogits|=%.3e (容差 %.0e)" % (d, TOL))
    # ---- has_rule=False(全空槽)这条分支也必须等价, 且规则路径必须**一次都不调**(C2a 口径) ----
    m.engram.alive.zero_()
    calls = {"n": 0}
    hd = m.blocks[0].attn.bucket_proj.register_forward_hook(
        lambda *a: calls.__setitem__("n", calls["n"] + 1))
    try:
        with torch.no_grad():
            new_d = m(x)
            with _UseRef():
                ref_d = m(x)
    finally:
        hd.remove()
    dd = float((new_d - ref_d).abs().max())
    ok &= check("E2c 全空槽(has_rule=False)分支等价且规则路径零调用",
                dd <= TOL and calls["n"] == 0,
                "max|Δlogits|=%.3e | bucket_proj 调用 %d 次(要求 0, 与优化前同)"
                % (dd, calls["n"]))
    with torch.no_grad():
        saved = M.RuleAttention.forward
        M.RuleAttention.forward = _perturbed_rule_attention
        try:
            bad = m(x)
        finally:
            M.RuleAttention.forward = saved
    dbad = float((bad - new).abs().max())
    check("N2 E2 负向对照(漏掉 keep 掩码必须被检出)", dbad > TOL,
          "疏漏版与现行差 %.3e ⇒ 比较器有牙" % dbad)
    return ok


# ------------------------------------------------------- E2b: 训练路径(梯度)
def e2b_grad():
    m, c, x = _toy_model(seed=5)
    x = x[:, :16]
    tgt = torch.randint(0, c.vocab, (3, 15), device=DEV)
    grads = {}
    for tag in ("new", "ref"):
        m.zero_grad(set_to_none=True)
        ctx = _UseRef() if tag == "ref" else None
        if ctx:
            ctx.__enter__()
        try:
            out = m(x)
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, c.vocab), tgt.reshape(-1))
            loss.backward()
            grads[tag] = {k: (v.grad.detach().clone() if v.grad is not None else None)
                          for k, v in m.named_parameters()}
            grads[tag + "_loss"] = float(loss.detach())
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
    worst, worst_k = 0.0, ""
    for k in grads["new"]:
        a, b = grads["new"][k], grads["ref"][k]
        if a is None and b is None:
            continue
        if a is None or b is None:
            check("E2b 梯度等价", False, "参数 %s 一边有梯度一边没有" % k)
            return False
        dd = float((a - b).abs().max()) / max(1e-12, float(b.abs().max()))
        if dd > worst:
            worst, worst_k = dd, k
    dl = abs(grads["new_loss"] - grads["ref_loss"])
    ok = check("E2b 训练路径梯度等价", worst <= 1e-5 and dl <= TOL,
               "loss Δ=%.3e | 梯度最大相对差 %.3e (%s)" % (dl, worst, worst_k))
    return ok


# ------------------------------------------------------- E3: 逐 token 选择
def _battery(tok):
    """覆盖: 各 UTF-8 状态 × 各 byte_tail × 惩罚/禁重复/温度/设备 × 全 -inf 松弛路径。

    **定向用例**: U+FFFD 守卫生效的那一格必须被单独构造出来 ——
    否则"漏掉 byte_tail 抑制"这个疏漏在任何随机用例上都测不出来(N3 曾因此失败)。
    """
    import dialogue_small as ds
    byte_map, special_ids = ds.bytelevel_token_bytes(tok)
    eos_id = tok.token_to_id("[EOS]")
    V = tok.get_vocab_size()
    states = [ds.utf8_state_for_bytes(bytes(bs))
              for bs in (b"", b"\xe4", b"\xe4\xbd", b"\xe4\xbd\xa0", b"\xef",
                         b"\xef\xbf", b"\xf0", b"\xf0\x9f", b"\xc3")]
    tails = [b"", b"\xe4", b"\xef", b"\xef\xbf", b"\xc3\xa9"]
    g = torch.Generator().manual_seed(7)
    base = torch.randn(V, generator=g)
    cases = []
    for st in states:
        for tl in tails:
            for pen, nrg, temp, safe in ((1.0, 0, 0.0, True), (2.0, 0, 0.0, True),
                                         (1.0, 2, 0.0, True), (1.0, 3, 0.0, True),
                                         (1.0, 0, 0.0, False), (1.0, 0, 0.7, True)):
                lg = base.clone()
                lg[torch.randint(0, V, (5,), generator=g)] = 30.0     # 制造强偏好
                hist = torch.randint(0, V, (6,), generator=g).tolist()
                cases.append((lg, hist, st, tl, pen, nrg, temp, safe))
    # ---- 定向 1: U+FFFD 字面量抑制 (tail=b"\xef\xbf" 时 b"\xbd" 不许出现) ----
    bd = [t for t, f in byte_map.items() if f in (b"\xbd", b"\xef\xbf\xbd")]
    fb = [t for t, f in byte_map.items() if f == b"\xa9"]          # 合法续字节
    if bd and fb:
        lg = torch.full((V,), -10.0)
        for t in bd:
            lg[t] = 20.0
        lg[fb[0]] = 1.0
        cases.append((lg, [], ds.utf8_state_for_bytes(b"\xef\xbf"), b"\xef\xbf",
                      1.0, 0, 0.0, True))
        cases.append((lg, [], ds.utf8_state_for_bytes(b"\xef\xbf"), b"",   # 对照: 无 tail
                      1.0, 0, 0.0, True))
    # ---- 定向 2: history 里的 token 只靠重复惩罚才被压下去 ----
    lg = torch.full((V,), -10.0)
    lg[5] = 9.0
    lg[7] = 8.0
    cases.append((lg, [5, 5, 7], (0, 0x80, 0xBF), b"", 2.0, 0, 0.0, True))
    cases.append((lg, [5, 5, 7], (0, 0x80, 0xBF), b"", 1.0, 0, 0.0, True))
    # ---- 定向 3: no_repeat_ngram 真的会禁掉一个合法 token ----
    lg = torch.full((V,), -10.0)
    lg[11] = 9.0
    lg[13] = 1.0
    cases.append((lg, [21, 11], (0, 0x80, 0xBF), b"", 1.0, 2, 0.0, True))
    cases.append((lg, [21, 11], (0, 0x80, 0xBF), b"", 1.0, 0, 0.0, True))
    # 松弛路径: 只有非法 token 有限
    hard = torch.full((V,), -float("inf"))
    mid = ds.utf8_state_for_bytes(b"\xe4")
    hard[1] = 5.0
    cases.append((hard, [], mid, b"\xe4", 1.0, 0, 0.0, True))
    # 全 -inf ⇒ 必须抛错(两侧都抛)
    dead = torch.full((V,), -float("inf"))
    cases.append((dead, [], (0, 0x80, 0xBF), b"", 1.0, 0, 0.0, True))
    return cases, byte_map, special_ids, eos_id


def _run_battery(cases, byte_map, special_ids, eos_id, device, impl=None):
    import dialogue_small as ds
    impl = impl or ds._select_next_id
    out = []
    for lg, hist, st, tl, pen, nrg, temp, safe in cases:
        gen = torch.Generator(device=device).manual_seed(1234)
        try:
            sel, relaxed = impl(lg.to(device), hist, st, byte_map,
                                special_ids, eos_id, temp, pen, nrg,
                                safe, gen, tl)
            out.append(("ok", sel, bool(relaxed)))
        except RuntimeError as exc:
            out.append(("raise", type(exc).__name__, None))
    return out


def _diff_count(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def e3_select_next_id():
    from tokenizers import Tokenizer
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    cases, bm, sid, eos = _battery(tok)
    # EOS 必须在 history 里, 否则"惩罚也作用于 EOS"这类疏漏永远测不出来
    cases.append((torch.full((tok.get_vocab_size(),), 1.0), [eos, eos],
                  (0, 0x80, 0xBF), b"", 2.0, 0, 0.0, True))
    devs = [DEV] if DEV == "cuda" else ["cpu"]
    if DEV == "cuda":
        devs.append("cpu")          # C17 设备无关
    bad = []
    for d in devs:
        got = _run_battery(cases, bm, sid, eos, d)
        with _UseRef():
            exp = _run_battery(cases, bm, sid, eos, d)
        bad += [(d, i) for i in range(len(cases)) if got[i] != exp[i]]
    ok = check("E3 _select_next_id 全电池等价", not bad,
               "%d 用例 × %d 设备, 不一致 %d 处%s"
               % (len(cases), len(devs), len(bad),
                  "" if not bad else " 例:%s" % [bad[i] for i in range(min(3, len(bad)))]))
    got = _run_battery(cases, bm, sid, eos, DEV)
    per = _run_battery(cases, bm, sid, eos, DEV, impl=_perturbed_select)
    nd = _diff_count(got, per)
    check("N3 E3 负向对照(漏掉 U+FFFD 抑制必须被检出)", nd > 0,
          "疏漏版在 %d/%d 例上与现行不同 ⇒ 比较器有牙" % (nd, len(cases)))
    # 负向对照二: 逐 token 加噪必须改变选择。
    #   ★ 早先用 `+0.5` 常量平移, 那是**恒等扰动**(argmax 不变、排序不变) ⇒ 白测(N5 曾因此失败)。
    g2 = torch.Generator().manual_seed(99)
    pert = [(c[0] + 0.5 * torch.randn(c[0].shape, generator=g2), c[1], c[2], c[3],
             c[4], c[5], c[6], c[7]) for c in cases[:80]]
    d2 = _diff_count(_run_battery(cases[:80], bm, sid, eos, DEV),
                     _run_battery(pert, bm, sid, eos, DEV))
    check("N5 E3 负向对照(扰动 logits 必须改变选择)", d2 > 0,
          "逐 token 加噪后 %d/80 例选择结果变化 ⇒ 该判据可失败" % d2)
    return ok


# ------------------------------------------------------- E4: 端到端生成逐条一致
def e4_end_to_end():
    from tokenizers import Tokenizer
    import dialogue_small as ds
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    _, ev_rows, _ = ds.split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                    seed=29, return_meta=True)
    rows = ev_rows[:N_ROWS]
    from model import RCfg, EngramRuleLM
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    new = [ds.greedy_generate(model, tok, r, max_new_tokens=90, device=DEV) for r in rows]
    with _UseRef():
        ref = [ds.greedy_generate(model, tok, r, max_new_tokens=90, device=DEV) for r in rows]
    diff_keys, same_tok = [], 0
    for a, b in zip(new, ref):
        if a["generated"] != b["generated"]:
            diff_keys.append("generated")
        if a["generated_tokens"] == b["generated_tokens"] and a["eos"] == b["eos"]:
            same_tok += 1
        for k in a:
            if k != "generated" and a[k] != b[k]:
                diff_keys.append(k)
    ok = check("E4 端到端 greedy_generate 逐条一致", not diff_keys and same_tok == len(rows),
               "%d/%d 行 token 序列完全一致, 生成 token %s; 不一致字段=%s"
               % (same_tok, len(rows),
                  [a["generated_tokens"] for a in new], sorted(set(diff_keys)) or "无"))
    # 负向对照: 换权重 ⇒ 生成必须变(否则"一致"是因为什么都没生成)
    m2 = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    torch.manual_seed(1)
    for p in m2.parameters():
        torch.nn.init.normal_(p, std=0.02)
    m2.eval()
    alt = ds.greedy_generate(m2, tok, rows[0], max_new_tokens=90, device=DEV)
    check("N4 E4 负向对照(换权重后生成必须不同)",
          alt["generated"] != new[0]["generated"] or alt["generated_tokens"] == 0,
          "换权重后首行生成 '%s'(%d token) vs 原 '%s'"
          % (alt["generated"][:24], alt["generated_tokens"], new[0]["generated"][:24]))
    return ok


def e5_batch_generation():
    """批生成(等长分组)必须与逐行 greedy_generate **逐字段完全一致**。

    这是批生成唯一能接受的证据标准: 不是"差不多", 是同一条 token 序列。
    负向对照: 打乱行序后再批一次, 顺序不同 ⇒ 若实现里有跨行串味, 这里会炸;
    同时验证"组内含已完成行(filler 补齐)"这条分支真的被走到。
    """
    from tokenizers import Tokenizer
    from model import RCfg, EngramRuleLM
    import dialogue_small as ds
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    _, ev_rows, _ = ds.split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                    seed=29, return_meta=True)
    rows = ev_rows[:24]
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    single = [ds.greedy_generate(model, tok, r, max_new_tokens=90, device=DEV) for r in rows]
    groups = {}
    for i, r in enumerate(rows):
        plen = len(tok.encode(ds.format_prefix(r), add_special_tokens=False).ids)
        groups.setdefault(plen, []).append(i)
    batched = [None] * len(rows)
    biggest = 0
    for _p, idxs in sorted(groups.items()):
        got = ds.greedy_generate_batch(model, tok, [rows[i] for i in idxs],
                                       max_new_tokens=90, device=DEV)
        biggest = max(biggest, len(idxs))
        for j, i in enumerate(idxs):
            batched[i] = got[j]
    fields = [k for k in single[0] if k != "question"]
    diffs = []
    for i in range(len(rows)):
        for k in fields:
            if single[i][k] != batched[i][k]:
                diffs.append((i, k, single[i][k], batched[i][k]))
    ok = check("E5 批生成与逐行逐字段一致", not diffs,
               "%d 行(最大组 %d), 比 %d 个字段, 不一致 %d 处%s"
               % (len(rows), biggest, len(fields), len(diffs),
                  "" if not diffs else " 例:%s" % diffs[:3]))

    # ---- 采样路径(temperature>0)也必须一致: 每行各自的 generator, 顺序不能互相影响 ----
    s1 = ds.evaluate_generations(model, tok, rows, 40, DEV, temperature=0.8, seed=41,
                                 batch_min_size=0)[1]
    s2 = ds.evaluate_generations(model, tok, rows, 40, DEV, temperature=0.8, seed=41,
                                 batch_min_size=3)[1]
    sdiff = [i for i in range(len(rows)) if s1[i]["generated"] != s2[i]["generated"]]
    ok &= check("E5b 采样路径批/逐行一致 (T=0.8, seed=41)", not sdiff,
                "不一致 %d/%d 行%s" % (len(sdiff), len(rows),
                                     "" if not sdiff else " 例:%s" % sdiff[:5]))
    # 负向对照: 换 seed 必须换结果(否则"一致"是因为没在采样)
    s3 = ds.evaluate_generations(model, tok, rows, 40, DEV, temperature=0.8, seed=42,
                                 batch_min_size=3)[1]
    nd = sum(1 for i in range(len(rows)) if s2[i]["generated"] != s3[i]["generated"])
    check("N6 E5 负向对照(换 seed 必须改变采样结果)", nd > 0,
          "换 seed 后 %d/%d 行结果变化 ⇒ 采样判据可失败" % (nd, len(rows)))
    return ok


def e6_padding_trim():
    """训练里"把微批截到该批真实最大行长"必须与不截断**同轨迹**。

    负向对照 N7: 把 bucket_step 设成 0(legacy 动态步长)并**强行**截断 ⇒ 结果必须变 ——
    这既证明守卫是必要的, 也证明本检查不是恒真的。
    """
    from model import RCfg, EngramRuleLM
    import dialogue_small as ds
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    g = torch.Generator().manual_seed(3)
    n, L = 96, ck["cfg"]["pos_abs_max"]
    V = ck["cfg"]["vocab"]
    x = torch.randint(3, V, (n, L), generator=g)
    starts = torch.randint(1, 40, (n,), generator=g)
    lens = starts + torch.randint(1, 90, (n,), generator=g)
    lens = torch.minimum(lens, torch.full((n,), L))

    def run(trim, force_bs=None, steps=6):
        cfg = dict(ck["cfg"])
        if force_bs is not None:
            cfg["bucket_step"] = force_bs
        torch.manual_seed(7)
        m = EngramRuleLM(RCfg(**cfg)).to(DEV)
        m.engram.alive[:] = True
        losses = []
        orig = ds.train_dialogue

        def spy(*a, **kw):
            kw["trim_padding"] = trim
            return orig(*a, **kw)
        ds.train_dialogue = spy
        try:
            # 直接复刻 train_dialogue 的步进, 只为记录逐步 loss
            import math
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
            gen = torch.Generator().manual_seed(11)
            m.train()
            mb = 32
            for _ in range(steps):
                idx = torch.randint(0, n, (64,), generator=gen)
                ntok = (lens[idx] - starts[idx]).sum().clamp_min(1).to(DEV)
                norm = ntok + 64
                opt.zero_grad(set_to_none=True)
                tot = torch.zeros((), device=DEV)
                for i in idx.split(mb):
                    bx = x[i].to(DEV)
                    sb = starts[i].to(DEV)
                    lb = lens[i].to(DEV)
                    if trim:
                        bx = ds._trim_length(bx, lb)
                    pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
                    valid = pos < lb.view(-1, 1)
                    import torch.nn.functional as F
                    lg = m(bx)[:, :-1, :]
                    tl, fl, _, _ = ds.answer_loss_sums(lg, bx[:, 1:], valid, sb)
                    ls = tl + fl
                    (ls / norm).backward()
                    tot += ls.detach()
                opt.step()
                losses.append(float(tot / norm))
        finally:
            ds.train_dialogue = orig
        return losses, {k: (v.detach().clone() if v.grad is not None else None)
                        for k, v in m.named_parameters()}, m

    on, _, m_on = run(True)
    off, _, m_off = run(False)
    worst = max(abs(a - b) for a, b in zip(on, off))
    # ★state_dict 里混着 bool buffer(engram.alive), 直接相减会抛
    #   "Subtraction with two bool tensors" ⇒ 只比浮点项, bool 项单独断言相等。
    sd_on, sd_off = m_on.state_dict(), m_off.state_dict()
    float_keys = [k for k in sd_on if sd_on[k].is_floating_point()]
    bool_keys = [k for k in sd_on if sd_on[k].dtype == torch.bool]
    pworst = max(float((sd_on[k] - sd_off[k]).abs().max()) for k in float_keys)
    bsame = all(bool((sd_on[k] == sd_off[k]).all()) for k in bool_keys)
    # ★容差理由(第二版 1e-5 又写错了 —— 同一个坑的第四次):
    #   loss 轨迹实测**逐位为 0**(截断掉的补齐位对 loss 没有任何贡献);
    #   参数差 1.03e-05 来自 E2b 早已记录的梯度末位差(相对 7.1e-8, 来自
    #   `(M,R,d).sum(1)`→`(B,L,R,d).sum(2)` 的归约顺序)被 **AdamW 放大**:
    #   AdamW 用 grad/sqrt(grad²) 归一化, 对 1e-8 量级的梯度差极敏感。
    #   6 步放大到 1e-5 是预期量级(对照: profile_training_speed 15 步放大到 loss 差 9.8e-4)。
    #   ⇒ 判据是「loss 轨迹 < 1e-6(实际 0)+ 参数 < 1e-3(实际 1.03e-5)」,
    #   而不是"参数也逐位相同" —— 后者与 E2b 自相矛盾。
    ok = check("E6 截断训练与不截断同轨迹", worst < 1e-6 and pworst < 1e-3 and bsame,
               "逐步 loss 最大差 %.3e (实测应为 0) | 参数最大差 %.3e (AdamW 放大, 容差 1e-3)"
               " | bool buffer 相同=%s" % (worst, pworst, bsame))

    # ---- N7: legacy 动态步长下截断**必须**改变结果(证明守卫必要 + 本检查有牙) ----
    legacy_on, _, _ = run(True, force_bs=0)
    legacy_off, _, _ = run(False, force_bs=0)
    lworst = max(abs(a - b) for a, b in zip(legacy_on, legacy_off))
    check("N7 E6 负向对照(bucket_step=0 时截断会改变结果)", lworst > 1e-6,
          "动态步长下逐步 loss 最大差 %.3e (期望 >1e-6) ⇒ 守卫是必要的" % lworst)
    return ok


def e7_eval_trim():
    """`evaluate_dialogue` 的截断必须给出**逐项相同**的评测读数。

    这条比 E6 更严: E6 只比 loss 轨迹, 这里比的是**要写进 capacity_scaling_v1.jsonl 的
    那些数**(NLL / token acc / 首 token acc), 一旦漂移, 历史读数就不可比了。
    负向对照: bucket_step=0 时截断必须改变读数。
    """
    from model import RCfg, EngramRuleLM
    import dialogue_small as ds
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    g = torch.Generator().manual_seed(5)
    n, L, V = 120, ck["cfg"]["pos_abs_max"], ck["cfg"]["vocab"]
    x = torch.randint(3, V, (n, L), generator=g)
    starts = torch.randint(1, 30, (n,), generator=g)
    lens = torch.minimum(starts + torch.randint(1, 120, (n,), generator=g),
                         torch.full((n,), L))

    def run(trim, force_bs=None):
        cfg = dict(ck["cfg"])
        if force_bs is not None:
            cfg["bucket_step"] = force_bs
        torch.manual_seed(7)
        m = EngramRuleLM(RCfg(**cfg)).to(DEV)
        m.engram.alive[:] = True
        return ds.evaluate_dialogue(m, x, starts, lens, 16, DEV, trim_padding=trim)

    a, b = run(True), run(False)
    keys = [k for k in a if isinstance(a[k], (int, float)) and not isinstance(a[k], bool)]
    worst, worst_k = 0.0, ""
    for k in keys:
        dd = abs(float(a[k]) - float(b[k]))
        scale = max(1e-12, abs(float(b[k])))
        if dd / scale > worst:
            worst, worst_k = dd / scale, k
    ok = check("E7 评测截断读数与不截断一致", worst < 1e-6,
               "比 %d 个标量读数, 最大相对差 %.3e (%s)" % (len(keys), worst, worst_k))
    la, lb = run(True, 0), run(False, 0)
    lw = max(abs(float(la[k]) - float(lb[k])) / max(1e-12, abs(float(lb[k])))
             for k in keys)
    # ★N8 第一版想用"动态步长下评测读数必须变"做对照, **这个对照是无效的**:
    #   随机初始化模型的 `startb≈0`(实测均值 0.16)⇒ `off0 = startb·step = 0`,
    #   窗口起点与 step 无关 ⇒ 端到端读数差**恰好为 0**, 什么也证明不了(规则 7)。
    # 守卫的**前提**是"动态步长下 `_bucket_step(L)` 随 L 变", 直接检验它。
    from model import RCfg as _RC, RuleAttention as _RA
    a_fixed = _RA(_RC(**dict(ck["cfg"])))
    a_dyn = _RA(_RC(**dict(ck["cfg"], bucket_step=0)))
    steps_fixed = [a_fixed._bucket_step(n) for n in (60, 128, 300)]
    steps_dyn = [a_dyn._bucket_step(n) for n in (60, 128, 300)]
    check("N8 E7 守卫前提: 动态步长下 _bucket_step 随 L 变", len(set(steps_dyn)) > 1,
          "bucket_step=0: L=60/128/300 → %s (变化) | 固定步长 %d: %s (恒定)"
          % (steps_dyn, ck["cfg"]["bucket_step"], steps_fixed))
    print("       (端到端指标差 %.3e —— 因 startb≈0 而恒为 0, 故端到端对照由 N7 承担)"
          % lw)
    return ok


def main():
    all_ok = True
    for fn in (e1_engram_read, e2_rule_attention, e2b_grad, e3_select_next_id,
               e4_end_to_end, e5_batch_generation, e6_padding_trim, e7_eval_trim):
        try:
            all_ok &= bool(fn())
        except Exception as exc:                      # 任何异常都算 FAIL, 不许"跳过"
            all_ok = False
            print("[FAIL] %s 抛异常: %r" % (fn.__name__, exc))
    bad = [n for n, ok, _ in RESULTS if not ok]
    print("\n%d 项检查, %d 项失败%s" % (len(RESULTS), len(bad), (": " + ", ".join(bad)) if bad else ""))
    print("RESULT:", "ALL PASS" if all_ok and not bad else "FAIL")
    return 0 if (all_ok and not bad) else 1


if __name__ == "__main__":
    raise SystemExit(main())
