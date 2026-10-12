"""§69.27 打通「带思维链的训练」：`seq=2048` + `think_start` 的最小可跑脚本。

**为什么需要它**：前面五级语料 + `answer_loss_sums(think_start=...)` + `pos_abs_max=2048`
三件都就绪了，但**从没有在同一次前向里跑通过**。
纪律：小步打通，先证明「梯度真的流到 think 段」再谈规模。

── 这一脚本验五件事 ────────────────────────────────────────────────
| 编号 | 验什么 | 判据 | 方向 |
|---|---|---|---|
| **T1** | `seq=2048` 下能装下带 think 的样本 | 不抛形状错 | — |
| **T2** | `think_start` 让 think 段进 loss | loss 变大 | ↑ 越大越好 |
| **T3** | ★think 段**真的在贡献梯度** | think 行梯度差 `max|Δg| > 0` | ↑ 非零即成立 |
| **N1** | ★T3 的负向对照（人为造反例） | 摘掉 think mask 后该差**恰好 0** | ↓ 必须归零 |
| **T4** | Engram 规则门是不是唯一入口 | alive 清零 ⇒ 与 `ablate_rule` 逐位相同 | 相同即成立 |
| **T5** | 可达距离（纯算术） | `n_buckets×bucket_step` 够 1M？ | 本次 **FAIL**（=512） |

★**T3 是关键**：T2 只证明 mask 变了，**不证明梯度真的流**。
  若实现里 mask 与 loss 的乘积分开了，mask 全 1 也可能零梯度。

★★**2026-10-11 修正**：T3 第一版失败（注意力/pos_abs/pos_bucket 梯度全零）。
  根因**不是** loss 实现有问题，而是 `model.py:497` 的
  `o = o * keep`（`keep = ~skip`）——`skip` 来自 `engram.read` 的 `~any_live`，
  而本脚本**直接构造模型、没建 Engram 表** ⇒ `alive` 全 False ⇒ `skip` 全 True
  ⇒ 窗口注意力输出被乘 0 ⇒ 该通路所有参数梯度**恒 0**。
  那是 `model.py:497` 的**设计**（§4.5 残差捷径），不是 bug；bug 在调用方。
  ★这个失败顺手挖出一条 1M 上下文的硬事实，见 T4。

用法：`python train_cot_smoke.py`
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

SEQ = 2048                      # ★§69.26 解锁的新 seq
PM = 2048                       # 位置表同步放大（§69.26）
THINK_WEIGHT = 0.6              # §69.23 的 A2 实测比值的倒数
ENGRAM_ROWS = 512               # 建 Engram 表用的样本行数（活着是判据的**输入**）
N_TEST = 8                      # 参与梯度判据的行数


def shipped_cfg():
    """出货 checkpoint 的**原始** cfg（未做任何覆盖）。"""
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    return dict(ck["cfg"])


def make_cfg():
    """本脚本统一的 cfg：出货 cfg + 本轮的 pos_abs_max=2048 覆盖。

    ★单一出处：T1b 与 T2/T3 必须用**同一份** cfg，否则两处读数不同口径
      （这正是 AGENTS 规则 11「检查的锚点必须与目标口径一致」的翻版）。
    ★同时返回出货的 pos_abs_max：T1b 的负向对照要拿它当基线档，
      而**不能**从已经覆盖过的 cfg 里读（第一版就这么读错了，两档都是 2048）。
    """
    cfg = shipped_cfg()
    ship_pm = int(cfg["pos_abs_max"])
    cfg["d"], cfg["n_heads"], cfg["n_slots"] = 64, 4, 4096
    cfg["pos_abs_max"] = PM
    return cfg, ship_pm


def encode_three(rows, tok, seq=SEQ):
    """把 (question, think, answer) 编成 **一条** 序列，并给出两段的起点。

    布局：`question + think + answer`，用 answer 的起点当 `starts`，
    think 的起点当 `think_start`。
    ★返回 (x, starts, think_start, lengths) —— 供 `answer_loss_sums` 直接消费。
    """
    xs, ss, ts, ls = [], [], [], []
    for r in rows:
        q_ids = tok.encode(r["question"], add_special_tokens=False).ids
        t_ids = tok.encode(r["think"], add_special_tokens=False).ids
        a_ids = tok.encode(r["answer"], add_special_tokens=False).ids
        ids = (q_ids + t_ids + a_ids)[:seq]
        ts.append(len(q_ids))                 # think 段起点
        ss.append(len(q_ids) + len(t_ids))    # answer 段起点
        ls.append(len(ids))
        xs.append(ids + [0] * (seq - len(ids)))
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ss), torch.tensor(ts), torch.tensor(ls))


def valid_from_lengths(lengths, seq=SEQ):
    """由行长造 `valid`（预测 x[i+1] 有意义 ⟺ i+1 ≤ length-1）。

    ★旧版用 `torch.ones(...)` 把补齐位也算进 loss —— 补齐位全是 token 0，
      既污染 loss 又稀释 think/answer 的真实占比。这里按真实行长开洞。
    """
    p = torch.arange(seq).view(1, -1)
    return p < (lengths.view(-1, 1) - 1)


def t1_fits():
    """T1：seq=2048 下形状合法，且 pm=2048 不越界。返回 (x,s,t,l, made)。"""
    print("【T1】seq=%d + pos_abs_max=%d 下能否装下带 think 的样本" % (SEQ, PM))
    E2.load_all()
    rows = E2.load_rows()
    import build_cot_corpus as B
    made, _st = B.build(rows[:2000], 2000)
    x, s, t, l = encode_three(made[:N_TEST], E2._TOK)
    print("  x 形状 =", tuple(x.shape), "（= (%d, %d)）" % (N_TEST, SEQ))
    print("  starts 范围 = %d ~ %d（必须 < %d）" % (int(s.min()), int(s.max()), PM))
    print("  think 起点范围 = %d ~ %d" % (int(t.min()), int(t.max())))
    print("  真实行长 = %d ~ %d（中位数 %d）"
          % (int(l.min()), int(l.max()), int(l.float().median())))
    assert x.shape == (N_TEST, SEQ), "T1 失败：形状不是 (%d, %d)" % (N_TEST, SEQ)
    assert int(s.max()) < PM, "T1 失败：answer 起点超过 pos_abs_max ⇒ 会 clamp 塌陷"
    assert int(t.max()) < int(s.min()), "T1 失败：think 起点 >= answer 起点"
    print("  ⇒ 形状合法、位置不越界、think 在 answer 之前 ⇒ T1 成立")
    return x, s, t, l, made


def t1b_long_row(cfg, ship_pm):
    """T1b：用**真的填满 2048 的长行**跑一次，验位置通路没塌。

    ★为什么必须有它：T1 那 8 行的真实行长只有 31~48 token（中位数 34），
      也就是说 `seq=2048` 里 **98% 是补齐位**，根本没测到长上下文位置通路。
      这一条拿拼卷语料（`target_tok=32768`）截一条真长的来跑。
    """
    import model as M
    p = os.path.join(HERE, "corpus_long_cot_packed.jsonl")
    with open(p, encoding="utf-8") as f:
        vol = json.loads(f.readline())
    text = "".join(s["text"] for s in vol["samples"])
    ids = E2._TOK.encode(text, add_special_tokens=False).ids[:SEQ]
    assert len(ids) == SEQ, \
        "T1b 前提失效：拼卷首卷不足 %d token（实得 %d），换一卷" % (SEQ, len(ids))

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor([ids], dtype=torch.long, device=DEV)

    # 位置编码的实际索引：model.py:419 / 472 都是 `pos.clamp(max=pos_abs_max-1)`
    pos = torch.arange(SEQ, device=DEV)
    print()
    print("【T1b】真长行（%d token，无补齐）下位置通路" % SEQ)
    rows_spec = [("出货 cfg  PM=%d" % ship_pm, ship_pm),
                 ("本轮 cfg  PM=%d" % PM, PM)]
    got = {}
    for label, pm in rows_spec:
        idx = pos.clamp(max=pm - 1)
        nd = int(torch.unique(idx).numel())
        got[pm] = nd
        print("  %-22s 实际用到的 pos_abs 行 = %5d / %d（%.1f%%）"
              % (label, nd, SEQ, 100.0 * nd / SEQ))
    assert ship_pm != PM, \
        "T1b 前提失效：出货 PM 与本轮 PM 相同（%d），负向对照无从建立" % PM

    # ★负向对照：判据看的量必须随 PM 变（规则 9）
    assert got[PM] == SEQ, "T1b 失败：PM=%d 时没有用满 %d 个位置行（实得 %d）" % (PM, SEQ, got[PM])
    assert got[PM] > got[ship_pm], \
        "T1b 负向对照失败：PM=%d 与 PM=%d 的位置行数相同（%d）⇒ 判据恒真" \
        % (PM, ship_pm, got[PM])
    print("  ⇒ 负向对照成立（PM 从 %d 改到 %d，位置行数 %d → %d，判据确实会变）"
          % (ship_pm, PM, got[ship_pm], got[PM]))
    print("  ⇒ 出货 cfg 下 %.1f%% 的位置**共享同一个位置编码**；PM=%d 才完全展开"
          % (100.0 * (1 - got[ship_pm] / SEQ), PM))

    # 真跑一次前向，证明这条长路走得通（形状/显存，不是质量）
    torch.manual_seed(11)
    m = M.EngramRuleLM(M.RCfg(**{**cfg, "pos_abs_max": PM})).to(DEV)
    m.eval()
    with torch.no_grad():
        y = m(x)
    print("  长行前向：logits %s，finite = %s" % (tuple(y.shape), bool(torch.isfinite(y).all())))
    assert y.shape[0] == 1 and y.shape[1] == SEQ and bool(torch.isfinite(y).all()), \
        "T1b 失败：长行前向形状异常或出现 NaN/Inf"
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    print("  ⇒ T1b 成立")
    return True


def t2_t3_loss_and_grad(made, cfg):
    """T2 + T3 + N1。返回 (model, cfg, res)。cfg 由 main 传入，保证与 T1b 同口径。"""
    import model as M
    x, s, t, l = encode_three(made[:N_TEST], E2._TOK)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(11)
    m = M.EngramRuleLM(M.RCfg(**cfg)).to(DEV)

    # ── ★建 Engram：不建则 alive 全 False ⇒ model.py:497 把注意力输出乘 0 ──
    ex, _es, _et, el = encode_three(made[:ENGRAM_ROWS], E2._TOK)
    info = E2.DS.build_engram_from_examples(m, ex, el)
    print("  建 Engram：%d 行 × seq=%d ⇒ alive %d / %d (%.1f%%)"
          % (ENGRAM_ROWS, SEQ, info["alive_slots"], info["n_slots"],
             100 * info["alive_ratio"]))
    # ★规则 29：写完对照必须断言「判据的输入真的变了」，否则是形式主义。
    assert info["alive_slots"] > 0, \
        "建表失败：alive 仍全 False ⇒ T3 必然零梯度（这正是第一版的失败原因）"
    m.train()
    valid = valid_from_lengths(l, SEQ)

    _orig_loss = E2.DS.answer_loss_sums

    def run(use_think, break_think=False):
        """跑一次前向+反传，返回 (loss, pos_abs.grad)。

        `break_think=True` 时注入反例：收下 `think_start` 却**不用它**
        （模拟「mask 与 loss 脱开」这个 T3 要抓的失败模式）。
        """
        f = _orig_loss
        if break_think:
            def f(logits, target, valid_, starts, think_start=None, think_weight=1.0):
                return _orig_loss(logits, target, valid_, starts)   # ★丢掉 think
        E2.DS.answer_loss_sums = f
        try:
            logits = m(x.to(DEV))
            lg, tgt = logits[:, :-1, :], x[:, 1:].to(DEV)
            kw = dict(think_start=t.to(DEV), think_weight=THINK_WEIGHT) if use_think else {}
            loss = E2.DS.answer_loss_sums(lg, tgt, valid.to(DEV), s.to(DEV), **kw)[0]
            m.zero_grad(set_to_none=True)
            loss.backward()
        finally:
            E2.DS.answer_loss_sums = _orig_loss      # ★必须复原
        pa = dict(m.named_parameters())["blocks.0.attn.pos_abs.weight"]
        return float(loss.detach()), pa.grad.detach().clone()

    print()
    print("【T2】loss 是否覆盖 think 段")
    l_no, g_no = run(False)
    l_yes, g_yes = run(True)
    print("  不带 think_start：loss = %.4f" % l_no)
    print("  带 think_start：  loss = %.4f（应更大）" % l_yes)
    assert l_yes > l_no, "T2 失败：带 think 的 loss 竟没更大 ⇒ mask 没生效"
    print("  ⇒ 增大 %.4f（+%.2f%%）⇒ T2 成立" % (l_yes - l_no, 100 * (l_yes / l_no - 1)))

    # think 段位置行 = 所有 (i, j) 的 j ∈ [think_start_i, answer_start_i)
    idx = sorted({j for i in range(x.shape[0])
                  for j in range(int(t[i]), int(s[i]))})

    print()
    print("【T3】think 段是否真的在贡献梯度（判据：think 行梯度差 > 0）")
    d_yes = float((g_yes[idx] - g_no[idx]).abs().max())
    print("  think 行（%d 行）带 think 时非零 = %d，|max| = %.4e"
          % (len(idx), int((g_yes[idx].abs() > 0).sum()), float(g_yes[idx].abs().max())))
    print("  ★主判据 max|g(带think) − g(不带think)| = %.6e  ⇒ 须 > 0" % d_yes)
    assert d_yes > 0.0, "T3 失败：带/不带 think 的梯度逐位相同 ⇒ think 段没在贡献"
    print("  ⇒ **think 段真的在贡献梯度** ⇒ T3 成立")

    print()
    print("【N1】★T3 的负向对照：人为把 think mask 摘掉（反例注入）")
    _l_brk, g_brk = run(True, break_think=True)
    d_brk = float((g_brk[idx] - g_no[idx]).abs().max())
    print("  注入反例后 loss = %.4f（应退回不带 think 的 %.4f）" % (_l_brk, l_no))
    print("  ★同一判据 max|g(注入) − g(不带think)| = %.6e  ⇒ 须**恰好 0**" % d_brk)
    assert _l_brk == l_no, "N1 失败：注入反例后 loss 没退回 %.6f（得 %.6f）" % (l_no, _l_brk)
    assert d_brk == 0.0, \
        "N1 失败：摘掉 think mask 后判据仍为 %.6e ⇒ T3 的判据恒真，不能用" % d_brk
    print("  ⇒ 判据在反例下确实归零 ⇒ T3 可失败，是真判据（规则 9/29）")

    nz_attn = sum(1 for n_, p_ in m.named_parameters()
                  if p_.grad is not None and int((p_.grad.abs() > 0).sum()) > 0
                  and n_.startswith("blocks.0.attn."))
    nz_engram = sum(1 for n_, p_ in m.named_parameters()
                    if p_.grad is not None and int((p_.grad.abs() > 0).sum()) > 0
                    and n_.startswith("engram."))
    print()
    print("  附带读数：blocks.0.attn.* 有梯度的参数 = %d 个；engram.* = %d 个"
          % (nz_attn, nz_engram))
    print("  （第一版这两个数都是 0 —— 那不是 loss 的锅，是没建 Engram 表）")
    res = dict(d_yes=d_yes, d_brk=d_brk, l_no=l_no, l_yes=l_yes,
               nz_attn=nz_attn, nz_engram=nz_engram,
               nz_think_rows=len(idx))
    return m, cfg, res


def t4_rule_gate(m, cfg, made):
    """T4：注意力通路**完全**被 Engram 规则门控，没有无条件/稠密通路。

    参照物 = `ablate_rule=True`（代码里把 `o` 与 `out_direct` 都清零的那条路）。
    · 反向：alive 清零 ⇒ 必须与参照物**逐位相同** ⇒ 硬门成立
    · 正向：alive 建好 ⇒ 必须与参照物**不同**   ⇒ 门确实开着
    两条都成立，才说明「规则是唯一入口」这个命题被真正测到。

    ★口径（2026-10-11 修正）：`model.py:500` 写的是
      `rule_skip_ratio_t = 1.0 - skip.float().mean()`
      ⇒ 它是「**没被跳过**的比例」，即**有多少位置拿得到注意力输出**，
      **不是**「被跳过的比例」。第一版我按后者读，断言方向整个反了。
      这次的方向由 `stats()` 的实现直接给出，不靠记忆。
    """
    DEV = next(m.parameters()).device
    x, _s, _t, _l = encode_three(made[:N_TEST], E2._TOK)
    x = x.to(DEV)
    m.eval()

    def fwd():
        with torch.no_grad():
            y = m(x).clone()
        return y, float(m.stats()["rule_skip_ratio"])

    alive_ratio = float(m.engram.alive.float().mean())

    # 参照：规则路径整个关掉（ablate_rule 会清零 o 与 out_direct）
    for b in m.blocks:
        b.attn.ablate_rule = True
    ref, _ = fwd()
    for b in m.blocks:
        b.attn.ablate_rule = False

    keep_alive = m.engram.alive.detach().clone()
    m.engram.alive.zero_()
    off, r_shut = fwd()
    m.engram.alive.copy_(keep_alive)
    on, r_open = fwd()

    d_off = float((off - ref).abs().max())
    d_on = float((on - ref).abs().max())

    print()
    print("【T4】Engram 规则门是不是唯一的注意力入口")
    print("  ★口径：rule_skip_ratio = 1 − mean(skip) = **拿得到注意力输出的位置比例**")
    print("  alive 存活率（%d 槽建自 %d 行） = %.4f" % (int(cfg["n_slots"]), ENGRAM_ROWS, alive_ratio))
    print("  rule_skip_ratio 开门(alive 已建) = %.6f ⇒ %.2f%% 的位置**有**注意力输出"
          % (r_open, 100 * r_open))
    print("  rule_skip_ratio 关门(alive 清零) = %.6f ⇒ %.2f%% 的位置**有**注意力输出"
          % (r_shut, 100 * r_shut))
    print("  max|logits(alive清零) − logits(关规则)| = %.6e  ⇒ 须 **恰好 0**" % d_off)
    print("  max|logits(alive建好) − logits(关规则)| = %.6e  ⇒ 须 **> 0**" % d_on)
    assert r_shut == 0.0, "T4 反例失败：alive 全 False 后仍有 %.4f 的位置拿到注意力输出" % r_shut
    assert d_off == 0.0, "T4 反例失败：alive 全 False 时与关规则仍有 %.6e 的差" % d_off
    assert d_on > 0.0, "T4 前提未达成：alive 建好后注意力仍然零贡献 ⇒ 表可能仍全空"
    print("  ⇒ alive 清零 ⇒ 与「关掉规则」**逐位相同** ⇒ ★model.py:497 的 `o = o * keep` 是**硬门**")
    print("     ⇒ 模型**没有**无条件/稠密注意力：Engram 不触发的位置，该层等价于恒等映射")
    print()
    print("  ★**但当前配置下这道门并不 binding**（这才是诚实的结论）：")
    print("     n_slots=%d / min_count=%d / 建表语料只有 %d 行 ⇒ 存活率 %.1f%%，"
          % (int(cfg["n_slots"]), int(cfg["min_count"]), ENGRAM_ROWS, 100 * alive_ratio))
    print("     实测 %.2f%% 的位置都触发到规则 ⇒ 「规则会不会触发」不是 1M 的瓶颈。" % (100 * r_open))
    print("     真正的瓶颈是**触发之后能看多远**（T5）与**窗口怎么采样**。")
    return r_open


def t5_reach(cfg):
    """T5：可达距离是**独立于 pos_abs_max** 的第二道上限，纯算术。

    ★off-by-one（2026-10-11 修正，由 audit_window_geometry.py 的 W1 抓出）：
      `start_bucket = bias_n.argmax(-1)`，桶下标只有 `0 .. n_buckets-1`，
      所以 `off0 = start_bucket*step` 的**最大���是 `(n_buckets-1)*step = 480`，
      不是 `n_buckets*step = 512`。⇒ 最远回看 = 480 + (W−1) = **495**，不是我先前写的 497。
    """
    nb, bs, W = int(cfg["n_buckets"]), int(cfg["bucket_step"]), int(cfg["window_max"])
    reach = nb * bs
    max_off0 = (nb - 1) * bs
    far = max_off0 + (W - 1)
    print()
    print("【T5】注意力可达距离（纯算术，不需要跑）")
    print("  n_buckets=%d × bucket_step=%d = %d token（名义值）" % (nb, bs, reach))
    print("  ★但 argmax 下标只有 0..%d ⇒ 最大 off0 = %d×%d = **%d**（不是 %d）"
          % (nb - 1, nb - 1, bs, max_off0, reach))
    print("  窗口是**连续** %d token ⇒ 最远被看到的位置 = %d + (%d−1) = **%d**"
          % (W, max_off0, W, far))
    assert far > 0, "T5 前提失效：算出的最远距离非正，本算术不适用"
    print("  ⇒ 当前 位置表 PM=%d、最远回看 %d，两者**都**远不到 1,000,000" % (PM, far))
    print("  ⇒ 放大 pos_abs_max **不会**让模型看得更远；两条通路必须分别改")
    print("  ⇒ 覆盖比例随 L 线性衰减：L=1M 时只覆盖 %.6f%%（见 audit_window_geometry.py）"
          % (100.0 * W / 1_000_000))
    return far


def main():
    print("=" * 74)
    print("带思维链训练的最小打通验证（seq=%d, pos_abs_max=%d）" % (SEQ, PM))
    print("=" * 74)
    _x, _s, _t, _l, made = t1_fits()
    cfg, ship_pm = make_cfg()
    t1b_long_row(cfg, ship_pm)
    m, cfg, res = t2_t3_loss_and_grad(made, cfg)
    r_open = t4_rule_gate(m, cfg, made)
    reach = t5_reach(cfg)
    print()
    print("=" * 74)
    print("【结论】")
    print("=" * 74)
    print("  ① 五级语料（§69.16/17/20/21/22）")
    print("  ② `answer_loss_sums(think_start=...)`（§69.23）")
    print("  ③ `pos_abs_max=2048`（§69.26）")
    print("  ⇒ think 段进 loss **且真的在贡献梯度**（T3 判据 %.4e；反例注入后恰好 %.1e）"
          % (res["d_yes"], res["d_brk"]))
    print()
    print("  ❗仍缺（按危害排序，均为本轮实测/实算，不是推测）：")
    print("     ① 可达距离只有 %d token（T5，纯算术 n_buckets×bucket_step）" % reach)
    print("        —— 与 pos_abs_max 是**两条独立**通路，放大位置表不会让它变远")
    print("     ② 窗口是**连续** %d token、不做稀疏/对数采样 ⇒ 即使起点到了远处，"
          % int(cfg["window_max"]))
    print("        也只覆盖那一小段连续区间，取不到远处那**一个**关键 token")
    print("     ③ 注意力**没有**无条件/稠密通路（T4：`o = o * keep` 是硬门）")
    print("        —— 但实测存活率 %.1f%% 时 %.2f%% 的位置都触发到规则，"
          % (100 * float(m.engram.alive.float().mean()), 100 * r_open))
    print("        所以这一条**当前不是瓶颈**；换小表/大语料后需要重测")
    print("     ④ 1M 位置编码本身")
    print()
    print("  ⇒ ①② 指向同一件事：现在的窗口是「**连续 16 格**」，不是「稀疏采样」；")
    print("     长程检索需要的是后者。要做 1M，先改窗口几何，不是先改位置表。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
