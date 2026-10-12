"""★预登记消融: 定位"问题→答案"的**绑定**步骤到底走哪条通路。

背景(本轮已实测, 见 DESIGN §39 / BRIEF §1.5⑥):
  在决定答案首 token 的位置(start−1):
    - Engram 检索键 = 问题末尾 2 个 token ⇒ 204 条留出问题 95.1% 检索到**相同的 4 个槽**
    - 窗口回看上限 15, 问题中位 19 token ⇒ 只有 29.4% 的问题被窗口完整覆盖
  而在答案中点, Engram 有 200/204 种不同组合 ⇒ Engram 对**续写**完全可区分。

问题: 绑定这一步, 到底是哪条通路在承载? 这决定了是该改 Engram 还是改窗口/内容通路。

────────────────────────────────────────────────────────────────────────
**预登记预测** (写在跑之前; 跑完只对照, 不改口径):

  P1  ablate_window 的损失在**首 token** 上 ≫ 在答案内部
      (绑定需要问题信息, 窗口是唯一覆盖问题的通路; 续写不需要)
  P2  ablate_direct 的损失在两个桶上都小 (direct 基本不承载信息)
  P3  ablate_rule(Engram 全关) 的首 token 损失 ≤ ablate_window 的首 token 损失
  P4  ★判决: 若 ablate_rule 与 ablate_window 的**首 token**损失接近
      ⇒ 首 token 完全由窗口承载 ⇒ "给 Engram 扩容/改检索键"是错的方向
  P5  若 ablate_rule 在首 token 上的损失**显著小于**它在答案内部的损失
      ⇒ Engram 对绑定无贡献、对续写有贡献 (与键恒定的测量一致)

────────────────────────────────────────────────────────────────────────
**判据可失败性** (AGENTS.md 规则 7) —— 三重自检, 任一不过就中止、不出数字:
  S1 重建正确性: 从 checkpoint 的 1,024 原始行重建的 eval 张量, 必须与 checkpoint
     自带的 eval_tokens **逐元素相同**。不同 ⇒ 重建错, 全部读数作废。
  S2 臂的有效性: 三个开关的 logits 必须与 full **不同**(否则该臂是空操作, 测的是 no-op)
  S3 臂的独立性: ablate_window 与 ablate_direct 的 logits 必须**互不相同**
     (否则两条腿没分开, 项目 C30 检查的同一件事)

**诚实边界**: 这是**推理期贡献归因**(在已训练的 v11 权重上关掉某条路),
不是重训消融。它回答"这条通路的输出对已学函数贡献多少", 不回答"如果训练时没有它会怎样"。
两条不要混。
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
SEQ = 512


def rebuild(ck):
    """从 checkpoint 自带原始行重建 split/tokenizer。返回 (train, eval) 三元组。"""
    from tokenizers import Tokenizer
    from dialogue_small import split_records, encode_examples
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    rows = ck["extra"]["raw_dialogue_rows"]
    tr_rows, ev_rows, _ = split_records(rows, train_fraction=0.8, seed=29, return_meta=True)
    train = encode_examples(tr_rows, tok, SEQ)
    ev = encode_examples(ev_rows, tok, SEQ)
    return train, ev


def set_arms(model, rule, window, direct):
    for blk in model.blocks:
        blk.attn.ablate_rule = rule
        blk.attn.ablate_window = window
        blk.attn.ablate_direct = direct


@torch.no_grad()
def logits_for(model, x, arm, bs=32):
    rule, window, direct = {
        "full": (False, False, False),
        "ablate_rule": (True, False, False),
        "ablate_window": (False, True, False),
        "ablate_direct": (False, False, True),
    }[arm]
    set_arms(model, rule, window, direct)
    outs = []
    for i in range(0, x.shape[0], bs):
        outs.append(model(x[i:i + bs]).float())
    return torch.cat(outs)


@torch.no_grad()
def measure(logits, x, starts, lengths):
    """分桶: 首 token (目标 x[start], 由位置 start−1 预测) vs 答案内部。"""
    B = x.shape[0]
    bi = torch.arange(B)
    first_pred = logits[bi, starts - 1].argmax(-1)
    first_hit = (first_pred == x[bi, starts]).float()
    nll_first = F.cross_entropy(logits[bi, starts - 1], x[bi, starts], reduction="none")

    pos = torch.arange(SEQ - 1).view(1, -1)
    tgt = x[:, 1:]
    m = pos >= (starts - 1).view(-1, 1)
    m &= pos < (lengths - 1).view(-1, 1)
    pred = logits[:, :-1].argmax(-1)
    hit = ((pred == tgt).float() * m).sum()
    nll = (F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           tgt.reshape(-1), reduction="none").view_as(tgt) * m).sum()
    return dict(first_acc=float(first_hit.mean()), first_nll=float(nll_first.mean()),
                rest_acc=float(hit / m.sum()), rest_nll=float(nll / m.sum()),
                n_first=int(B), n_rest=int(m.sum()))


def main():
    from model import RCfg, EngramRuleLM

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    (tr_x, tr_s, tr_l), (ev_x, ev_s, ev_l) = rebuild(ck)

    # ---- S1 重建正确性 ----
    ck_ev = ck["eval_tokens"]
    ck_st = torch.tensor(ck["extra"]["eval_answer_starts"], dtype=torch.long)
    ok_shape = (ev_x.shape == ck_ev.shape)
    ok_vals = bool((ev_x == ck_ev).all()) if ok_shape else False
    ok_starts = bool((ev_s == ck_st).all())
    print("[S1] 重建校验: shape %s / 逐元素相同 %s / answer_start 相同 %s  (train %d, eval %d)"
          % (ok_shape, ok_vals, ok_starts, tr_x.shape[0], ev_x.shape[0]))
    if not (ok_shape and ok_vals and ok_starts):
        print("     ★自检失败: 重建与 checkpoint 不一致, 按纪律中止, 不出任何读数。")
        return 2
    print("     → 自检通过, 继续。")

    model = EngramRuleLM(RCfg(**ck["cfg"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # ---- S2/S3 臂的有效性与独立性 (只在 eval 上验证, 便宜) ----
    lg = {a: logits_for(model, ev_x, a) for a in
          ("full", "ablate_rule", "ablate_window", "ablate_direct")}
    print("\n[S2] 各臂与 full 的 logits 最大绝对差 (==0 说明该臂是空操作):")
    for a in ("ablate_rule", "ablate_window", "ablate_direct"):
        print("     %-14s %.6f" % (a, float((lg[a] - lg["full"]).abs().max())))
    print("[S3] 臂间独立性:")
    print("     window vs direct %.6f" % float((lg["ablate_window"] - lg["ablate_direct"]).abs().max()))
    live = all(float((lg[a] - lg["full"]).abs().max()) > 1e-4 for a in
               ("ablate_rule", "ablate_window", "ablate_direct"))
    indep = float((lg["ablate_window"] - lg["ablate_direct"]).abs().max()) > 1e-4
    if not (live and indep):
        print("     ★自检失败: 有空操作臂或两臂未分开, 中止。live=%s indep=%s" % (live, indep))
        return 2
    print("     → 自检通过。")

    # ---- 正式读数 ----
    res = {}
    for split, (x, s, l) in (("eval", (ev_x, ev_s, ev_l)), ("train", (tr_x, tr_s, tr_l))):
        print("\n=== %s (n=%d) ===" % (split, x.shape[0]))
        print("  %-14s %10s %10s %10s %10s" % ("arm", "首token acc", "首token NLL", "内部 acc", "内部 NLL"))
        base = None
        for a in ("full", "ablate_rule", "ablate_window", "ablate_direct"):
            m = measure(logits_for(model, x, a), x, s, l)
            if a == "full":
                base = m
                print("  %-14s %10.4f %10.4f %10.4f %10.4f" % (a, m["first_acc"], m["first_nll"], m["rest_acc"], m["rest_nll"]))
            else:
                res[(split, a)] = m
                print("  %-14s %10.4f %10.4f %10.4f %10.4f"
                      % (a, m["first_acc"], m["first_nll"], m["rest_acc"], m["rest_nll"]))
        for a in ("ablate_rule", "ablate_window", "ablate_direct"):
            m = res[(split, a)]
            print("     %-14s 相对损失: 首token %+.4f  答案内部 %+.4f"
                  % (a, m["first_acc"] - base["first_acc"], m["rest_acc"] - base["rest_acc"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())