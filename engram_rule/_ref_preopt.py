"""★ 冻结参照: 2026-10-10 性能优化**之前**的三段热路径, 逐字拷贝, 之后不许再改。

存在的唯一目的: 让 `check_fast_path_equiv.py` 能拿**真实的旧代码**当金标准,
而不是让检查脚本里再抄一遍(抄一遍就有"抄错了两处而互相抵消"的风险 ——
本项目已经吃过三次"检查本身写错"的亏, AGENTS.md 规则 5/7)。

三段热路径(优化前):
  1. EngramMemory.read        —— 逐位置布尔掩码索引(每次 nonzero ⇒ 一次 device→host 同步)
  2. RuleAttention.forward    —— 每层 int(keep.sum()) + 5 处布尔掩码索引
  3. _select_next_id          —— 每步重建 allowed 列表 + vocab 次逐元素张量赋值

被参照的现行实现:
  model.py: EngramMemory.read / RuleAttention.forward
  dialogue_small.py: _select_next_id
"""
import torch

from dialogue_small import valid_token_bytes


# ---------------------------------------------------------------- 1) Engram 软读
def ref_engram_read(self, x, emb):
    c = self.c
    slot = self.slot_ids(x)
    B, L, R = slot.shape
    live = self.alive[slot]
    any_live = live.any(-1)
    flat = any_live.reshape(-1)
    v_out = torch.zeros(B, L, R, c.d, device=x.device, dtype=self.val.weight.dtype)
    w_out = torch.zeros(B, L, R, device=x.device, dtype=self.val.weight.dtype)
    if bool(flat.any()):
        sel = slot.reshape(-1, R)[flat]
        idx = flat.nonzero(as_tuple=True)[0]
        bi, ti = idx // L, idx % L
        q = self.ngram_query(x, emb, bi, ti)
        k = self.key(sel)
        v = self.val(sel)
        m_live = live.reshape(-1, R)[flat]
        logits = -c.beta * (q - k).pow(2).sum(-1)
        logits = logits.masked_fill(~m_live, float("-inf"))
        att = torch.nan_to_num(logits.softmax(-1), nan=0.0)
        v_out.reshape(-1, R, c.d)[flat] = v
        w_out.reshape(-1, R)[flat] = att
    return v_out, w_out, ~any_live


# ---------------------------------------------------------------- 2) 规则注意力
def ref_rule_attention_forward(self, h, pos, rule_vec, rule_w, skip, causal=True, has_rule=None):
    from model import rope
    c = self.c
    B, L, _ = h.shape
    q = rope(self.q(h).view(B, L, c.n_heads, c.head_dim), pos, c.rope_dim)
    k = rope(self.k(h).view(B, L, c.n_heads, c.head_dim), pos, c.rope_dim)
    v = self.v(h).view(B, L, c.n_heads, c.head_dim)

    keep = ~skip
    bias = torch.zeros(B, L, c.n_buckets, device=h.device, dtype=h.dtype)
    n_kept = int(keep.sum())
    out_direct = torch.zeros(B, L, c.d, device=h.device, dtype=h.dtype)
    if n_kept > 0:
        rv, wt = rule_vec[keep], rule_w[keep]
        g = torch.sigmoid(self.gate * rv)
        hw = self.head_w.view(1, -1, 1)
        agg = (hw * g * rv * wt.unsqueeze(-1)).sum(1)
        bias_kept = self.bucket_proj(agg)
        if c.use_pos_bucket:
            pi = pos.clamp(max=c.pos_abs_max - 1).view(1, L).expand(B, L)
            bias_kept = bias_kept + self.pos_bucket(pi[keep]).to(bias_kept.dtype)
        bias[keep] = bias_kept.to(bias.dtype)
        self.last_agg, self.last_bias = agg.detach(), bias_kept.detach()
        out_direct[keep] = self.direct(agg).to(h.dtype)
    else:
        self.last_agg, self.last_bias = None, None
    if self.ablate_rule:
        bias = torch.zeros_like(bias)
        out_direct = torch.zeros_like(out_direct)
    if self.ablate_direct:
        out_direct = torch.zeros_like(out_direct)

    bias_n = (bias - bias.mean(-1, keepdim=True)) / (bias.std(-1, keepdim=True) + 1e-5)
    bias_n = bias_n * self.rule_scale
    startb = bias_n.argmax(-1)
    self.last_startb = startb.detach() if self.record else None
    jidx, jvalid, off = self._window_index(startb, L, pos)
    W = jidx.shape[-1]
    self.last_window = W
    self.last_window_off = off.detach() if self.record else None
    bidx = (off // self._bucket_step(L)).clamp(0, c.n_buckets - 1)
    bsel = bias_n.gather(-1, bidx)

    flat_k = k.reshape(B, L, c.n_heads * c.head_dim)
    bsel_b = torch.arange(B, device=h.device).view(B, 1, 1)
    kk = flat_k[bsel_b, jidx].reshape(B, L, W, c.n_heads, c.head_dim)
    vv = v.reshape(B, L, c.n_heads * c.head_dim)[bsel_b, jidx].reshape(B, L, W, c.n_heads, c.head_dim)
    vmask = jvalid.view(B, L, W, 1, 1).to(h.dtype)

    gc = torch.sigmoid(self.content_gate(h))
    self.last_gc_t = gc.float().mean() if self.record_gc else None
    self.last_gc = float(self.last_gc_t) if self.record_gc else None
    gq = self.wq(q)
    if c.use_pos_abs:
        gq = gq + self.pos_abs(pos.clamp(max=c.pos_abs_max - 1)).unsqueeze(1)
    gq = gq.unsqueeze(2) * (gc * self.lam).view(B, L, 1, 1, 1)
    bterm = bsel
    gate = torch.nn.functional.silu(self.wg(kk) + gq + bterm.view(B, L, W, 1, 1))
    up = self.wu(vv)
    if c.local_norm:
        gsum = (gate.abs() * vmask).sum(2)
        o = (gate * up * vmask).sum(2) / (gsum + 1e-5)
    else:
        o = (gate * up * vmask).sum(2) / max(W, 1) ** 0.5
    if self.record:
        self.last_contrib = (gate * up * vmask).sum(dim=(3, 4)).detach()
    o = self.wd(o).reshape(B, L, c.n_heads * c.head_dim)
    if self.ablate_window or self.ablate_rule:
        o = torch.zeros_like(o)
    if self.record:
        self.last_mag = dict(rule_absmean=float(bias_n.abs().mean()),
                             window=W, gate_mean=float(gate.mean()),
                             rule_scale=float(self.rule_scale), lam=float(self.lam),
                             head_w_mean=float(self.head_w.mean()))
    o = o * keep.unsqueeze(-1).to(o.dtype)
    self.last_stats = dict(kept=n_kept, total=B * L,
                           rule_skip_ratio=1.0 - n_kept / max(B * L, 1))
    return self.o(o) + out_direct


# ---------------------------------------------------------------- 3) 逐 token 选择
def ref_select_next_id(logits, history, byte_state, byte_map, special_ids,
                       eos_id, temperature, repetition_penalty, no_repeat_ngram,
                       utf8_safe=True, generator=None, byte_tail=b"", mask_cache=None):
    """mask_cache 是 §51 之后新增的第 13 个位置参数(查表对象)。参照实现**故意忽略**它 ——
    它只是把"每步重算 allowed"换成了"查表", 不参与选择语义。"""
    scores = logits.float().clone()
    scores[~torch.isfinite(scores)] = -float("inf")
    if utf8_safe:
        allowed = []
        for token_id, fragment in byte_map.items():
            if fragment is None:
                if token_id == eos_id and byte_state == (0, 0x80, 0xBF):
                    allowed.append(token_id)
            elif (valid_token_bytes(byte_state, fragment) is not None and
                  b"\xef\xbf\xbd" not in (byte_tail + fragment)):
                allowed.append(token_id)
    else:
        allowed = list(range(scores.numel()))
    if not allowed:
        raise RuntimeError("decoder constraints left no legal next token")
    allowed_set = set(allowed)
    for token_id in range(scores.numel()):
        if token_id not in allowed_set:
            scores[token_id] = -float("inf")
    if repetition_penalty > 1.0:
        for token_id in set(history):
            if token_id != eos_id and torch.isfinite(scores[token_id]):
                scores[token_id] = (scores[token_id] / repetition_penalty if scores[token_id] > 0
                                    else scores[token_id] * repetition_penalty)
    if no_repeat_ngram > 0 and len(history) >= no_repeat_ngram - 1:
        prefix = tuple(history[-(no_repeat_ngram - 1):]) if no_repeat_ngram > 1 else ()
        for i in range(len(history) - no_repeat_ngram + 1):
            gram = history[i:i + no_repeat_ngram]
            if tuple(gram[:-1]) == prefix and gram[-1] in allowed_set and gram[-1] != eos_id:
                scores[gram[-1]] = -float("inf")
    constraint_relaxed = not bool(torch.isfinite(scores).any())
    if constraint_relaxed:
        scores = logits.float().clone()
        for token_id in range(scores.numel()):
            if token_id not in allowed_set:
                scores[token_id] = -float("inf")
        if repetition_penalty > 1.0:
            for token_id in set(history):
                if token_id != eos_id and torch.isfinite(scores[token_id]):
                    scores[token_id] = (scores[token_id] / repetition_penalty if scores[token_id] > 0
                                        else scores[token_id] * repetition_penalty)
    if not bool(torch.isfinite(scores).any()):
        raise RuntimeError("all legal decoder candidates have non-finite logits")
    if temperature == 0:
        return int(scores.argmax()), constraint_relaxed
    probs = torch.softmax(scores / temperature, dim=-1)
    if not bool(torch.isfinite(probs).all()) or float(probs.sum()) <= 0:
        raise RuntimeError("generation sampler produced invalid probabilities")
    return int(torch.multinomial(probs, num_samples=1, generator=generator)), constraint_relaxed
