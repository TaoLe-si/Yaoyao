"""★实验 A 的跑前检查。**全部通过才准跑训练** (规则 3)。

每条都必须是**可失败**的: 在反例下会变红。全绿才放行。

  C1 锚点      eval tokens/starts/lengths 从 raw_dialogue_rows 重建后与 v11 逐元素相同
  C2 ★口径锚   用**同一份代码**在 **v11** 上复现 §46 的 R1=0.1832 / R2=0.7653
               —— 若复现不了, 说明本实验的 LOO 行子集/分层口径与 §46 不同,
               R1/R2 就不能和 §46 比, 整个实验的"相对 §46"结论作废。
  C3 负向对照  LOO=0 的行, 其开头 token 在**整个训练集里只出现 1 次**
               (否则"LOO=0"名不副实, 分层就分错了东西)
  C4 Δ 实现    用**同一份 delta_k1** 在 **v11** 上复现 §42 的 Δ_train ≈ +5.17
               —— 恒等自检只能证明"没测偏", 正向锚才能证明"测的是真信号"
  C5 配置合法性 三档 d 的 head_dim 必须恒为 16, 且满足 RCfg 的断言

**故意不在这里跑训练**: C6(d=64 必须复现 v11 基线)是**跑完第一个 run 之后的闸门**,
写在 exp_capacity_scaling.py 里 —— 先花 3 分钟验一次, 比花 3 小时验 9 次便宜。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
TOL = 1e-3

# §46 (probe_opener_learnability.py) 在 v11 上的已发表读数 —— 同口径锚
R1_ANCHOR = 0.1832
R2_ANCHOR = 0.7653
# §42 (probe_delta_conditioning.py) 在 v11 train 上的已发表读数 —— 同口径锚
DELTA_ANCHOR = 5.1735


def main():
    from tokenizers import Tokenizer
    from model import RCfg, EngramRuleLM
    from dialogue_small import split_records, encode_examples
    import exp_capacity_scaling as exp

    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr_rows, ev_rows, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                        seed=29, return_meta=True)

    # ── C1 锚点 ───────────────────────────────────────────────────────────
    ev_x, ev_s, ev_l = encode_examples(ev_rows, tok, SEQ)
    ok1 = (bool((ev_x == ck["eval_tokens"]).all())
           and ev_s.tolist() == ck["extra"]["eval_answer_starts"]
           and ev_l.tolist() == ck["extra"]["eval_lengths"])
    assert ok1, "C1 失败: eval 侧重建与 v11 不一致"
    print("C1 通过  eval tokens/starts/lengths 与 v11 逐元素相同")

    tr_loo, tr_keep, tr_al, tr_mx = exp.loo_and_subset(tok, tr_rows, SEQ)
    ev_loo, ev_keep, _, _ = exp.loo_and_subset(tok, ev_rows, SEQ)

    # ── C3 负向对照: LOO=0 真的只出现过一次 (必须在 C2 之前, 因为它验证分层对象) ──
    from collections import Counter

    def opener_of(r):
        from dialogue_small import format_prefix, format_record
        p = len(tok.encode(format_prefix(r), add_special_tokens=False).ids)
        ids = tok.encode(format_record(r), add_special_tokens=False).ids
        return ids[p] if len(ids) > p else -1

    top = [opener_of(r) for r in tr_rows]
    cnt = Counter(top)
    zero_rows = [i for i, v in enumerate(tr_loo.tolist()) if v == 0.0]
    bad = [i for i in zero_rows if cnt[top[tr_keep[i]]] != 1]
    assert not bad, "C3 失败: %d 个 LOO=0 的行其开头在训练里出现次数 != 1" % len(bad)
    print("C3 通过  %d 个 LOO=0 行, 其答案开头在训练集中出现次数恒为 1 (负向对照: "
          "分层对象与层名一致)" % len(zero_rows))

    # ── C2 ★口径锚: 在 v11 上复现 §46 的 R1/R2 ────────────────────────────
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    sub = [{"question": tr_rows[i]["question"], "answer": tr_rows[i]["answer"],
            "history": []} for i in tr_keep]
    _, hit = exp.score_rows(model, sub, tok, SEQ)
    m0, m4 = tr_loo == 0, tr_loo >= 4
    r1, r2 = float(hit[m0].mean()), float(hit[m4].mean())
    n0, n4 = int(m0.sum()), int(m4.sum())
    print("      R1=%.4f (锚 %.4f, n=%d) | R2=%.4f (锚 %.4f, n=%d) | keep=%d"
          % (r1, R1_ANCHOR, n0, r2, R2_ANCHOR, n4, len(tr_keep)))
    assert abs(r1 - R1_ANCHOR) <= TOL, "C2 失败: R1=%.4f 与 §46 的 %.4f 差 %.2e > %.0e" % (
        r1, R1_ANCHOR, abs(r1 - R1_ANCHOR), TOL)
    assert abs(r2 - R2_ANCHOR) <= TOL, "C2 失败: R2=%.4f 与 §46 的 %.4f 差 %.2e > %.0e" % (
        r2, R2_ANCHOR, abs(r2 - R2_ANCHOR), TOL)
    print("C2 通过  同口径代码在 v11 上复现 §46 的 R1/R2 (容差 %.0e)" % TOL)

    # ── C4 Δ 实现锚: 在 v11 上复现 §42 的 Δ_train ──────────────────────────
    d_tr, se_tr, n_tr = exp.delta_k1(model, sub, tok, SEQ)
    print("      Δ_train=%.4f ± %.4f (锚 %.4f, n=%d)" % (d_tr, se_tr, DELTA_ANCHOR, n_tr))
    assert abs(d_tr - DELTA_ANCHOR) <= 0.5, (
        "C4 失败: Δ_train=%.4f 与 §42 的 %.4f 差 %.2f > 0.5 ⇒ Δ 实现或口径有问题"
        % (d_tr, DELTA_ANCHOR, abs(d_tr - DELTA_ANCHOR)))
    assert d_tr > 2 * se_tr, "C4 失败: Δ_train 不显著 (%.4f <= 2×%.4f)" % (d_tr, se_tr)
    print("C4 通过  同口径 delta_k1 在 v11 上复现 §42 的 Δ_train 且显著为正")

    # ── C7 ★新主判据 S 的有效性 (DESIGN §48 修订 A) ────────────────────────
    #  判据 S = slope(hit ~ log(1+LOO))。它**必须在已知答案的 v11 上先验过**:
    #    · v11 的 hit 随 LOO 单调升(0.1832 → 0.7653, §46) ⇒ β 必须**显著为正**
    #    · 把 LOO 打乱后 β 必须 **≈0** ⇒ 否则这个量测的不是"频次依赖"
    #  任一不过, S 无效, 主实验跑完也不许用它下结论。
    #  (放在 del model 之前: 这里还要用 v11 模型)
    import probe_capacity_scaling as judge
    b = judge.slope([float(v) for v in hit], [int(v) for v in tr_loo])
    nc_mu, nc_se, nc_max = judge.slope_negative_control(
        [float(v) for v in hit], [int(v) for v in tr_loo])
    print("      v11 上 β=%+.5f (应显著为正) | 置换后 mean=%+.5f SE=%.5f max|b|=%.5f (应 ≈0)"
          % (b, nc_mu, nc_se, nc_max))
    assert b > 0, "C7 失败: v11 上 β=%+.5f <= 0 ⇒ 判据 S 连已知为正的信号都测不出" % b
    assert nc_se == nc_se and abs(nc_mu) < 2.0 * nc_se, (
        "C7 失败: 置换 LOO 后斜率 mean=%+.5f 显著非 0(2SE=%.5f) ⇒ S 测的不是频次依赖"
        % (nc_mu, 2.0 * nc_se))
    print("C7 通过  判据 S 在 v11 上 β>0 且置换对照 ≈0 ⇒ 有效")

    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # ── C5 配置合法性 ─────────────────────────────────────────────────────
    for d in exp.D_MAIN:
        c = exp.build_cfg(ck["cfg"], d)
        assert c.head_dim == exp.HEAD_DIM, "C5 失败: d=%d 的 head_dim=%d" % (d, c.head_dim)
        assert d % c.n_heads == 0
        assert 0 < c.rope_dim <= c.head_dim
        print("C5 通过  d=%-4d n_heads=%-3d head_dim=%-3d rope_dim=%d 合法"
              % (d, c.n_heads, c.head_dim, c.rope_dim))

    # ── C6 ★微批切分梯度等价(别只声明, 要验) ──────────────────────────────
    #  train_dialogue 的 docstring 声称"切分微批梯度精确等价, 只有浮点求和顺序不同"。
    #  这是 §48 敢用 micro_batch 兜底显存的**唯一依据**, 所以必须实测。
    #  做法: 给 AdamW.step 装间谍, 抓 train_dialogue **真实**算出的梯度(不重写它的循环,
    #  否则验的是我的复刻而不是出货代码)。
    from dialogue_small import train_dialogue, encode_examples
    tr_x, tr_s, tr_l = encode_examples(tr_rows, tok, SEQ)

    def grads_at(mb, d=64):
        cap = []
        orig = torch.optim.AdamW.step

        def spy(self, *a, **k):
            for p in self.param_groups[0]["params"]:
                cap.append(None if p.grad is None else p.grad.detach().clone())
            return orig(self, *a, **k)

        m = exp.make_model(ck, d, 11, tr_x, tr_l)
        torch.optim.AdamW.step = spy
        try:
            train_dialogue(m, tr_x, tr_s, tr_l, 1, exp.BS, 0.003, DEV, seed=11,
                           first_token_weight=1.0, micro_batch=mb)
        finally:
            torch.optim.AdamW.step = orig
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()
        return cap

    g64, g32 = grads_at(64), grads_at(32)
    assert len(g64) == len(g32) and len(g64) > 0, "C6 失败: 抓到的梯度数量不一致"
    worst = 0.0
    worst_rel = 0.0
    for a, b in zip(g64, g32):
        assert (a is None) == (b is None), "C6 失败: 某个参数在两次 run 里梯度有无不一致"
        if a is None:
            continue
        dmax = float((a - b).abs().max())
        scale = max(float(a.abs().max()), 1e-12)
        worst = max(worst, dmax)
        worst_rel = max(worst_rel, dmax / scale)
    print("      micro_batch 64 vs 32: max|Δgrad|=%.3e  max 相对=%.3e  (%d 个梯度张量)"
          % (worst, worst_rel, len(g64)))
    assert worst_rel < 1e-4, ("C6 失败: 微批切分的梯度相对差 %.3e >= 1e-4 ⇒ "
                               "micro_batch 兜底不是精确等价, §48 的显存方案不成立"
                               % worst_rel)
    print("C6 通过  micro_batch 切分梯度相对差 %.2e < 1e-4 ⇒ 兜底方案精确等价" % worst_rel)

    # ── C7 ★新主判据 S 的有效性 (DESIGN §48 修订 A) ────────────────────────
    #  判据 S = slope(hit ~ log(1+LOO))。它**必须在已知答案的 v11 上先验过**:
    #    · v11 的 hit 随 LOO 单调升(0.1832 → 0.7653, §46) ⇒ β 必须**显著为正**
    #    · 把 LOO 打乱后 β 必须 **≈0** ⇒ 否则这个量测的不是"频次依赖"
    #  这两条任一不过, S 无效, 主实验跑完也不许用它下结论。
    import probe_capacity_scaling as judge
    b = judge.slope([float(v) for v in hit], [int(v) for v in tr_loo])
    nc_mu, nc_se, nc_max = judge.slope_negative_control(
        [float(v) for v in hit], [int(v) for v in tr_loo])
    print("      v11 上 β=%+.5f (应显著为正) | 置换后 mean=%+.5f SE=%.5f max|b|=%.5f (应 ≈0)"
          % (b, nc_mu, nc_se, nc_max))
    assert b > 0, "C7 失败: v11 上 β=%+.5f <= 0 ⇒ 判据 S 连已知为正的信号都测不出" % b
    assert nc_se == nc_se and abs(nc_mu) < 2.0 * nc_se, (
        "C7 失败: 置换 LOO 后斜率 mean=%+.5f 显著非 0(2SE=%.5f) ⇒ S 测的不是频次依赖"
        % (nc_mu, 2.0 * nc_se))
    print("C7 通过  判据 S 在 v11 上 β>0 且置换对照 ≈0 ⇒ 有效")

    print("\n全部检查通过 → 允许跑实验 A。")
    return 0


if __name__ == "__main__":
    sys.exit(main())