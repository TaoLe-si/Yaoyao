"""★预登记: 把"首 token 在训练里没当过开头"这 13.8% 再拆成【角色】与【词项】。

§44 实测: 首答案 token 在训练中**作为首 token 出现频次 = 0** 的那 28 条(13.8%),
Δ_i 在四个距离层**全部为负** —— 模型系统性更偏好错配答案的首 token。
但那一组把两件事混在一起了, 本脚本拆开。

────────────────────────────────────────────────────────────────────────
**★分层的精确定义(比"是否当过开头"更严格)**

模型有**两份独立参数**:
  · `emb.weight[t]`   —— 只要 t 出现在任何输入位置就收梯度
  · `head.weight[t]`  —— 只有 t 作为**答案里的目标 token** 才收梯度(loss 只监督答案)

所以真正决定"能不能输出 t"的是**它有没有在训练答案里出现过**。据此分三层:

  **A 组** t 在训练答案里出现过 **且** 当过答案首 token  ⇒ head 受训 + 首 token 角色受训
  **B 组** t 在训练答案里出现过(任意位置), 但**从未当过答案首 token** ⇒ head 受训, 角色未受训
         ⇒ 候选: **角色问题**(模型没被训练过"用这个字开头")
  **C 组** t **从未在训练答案里出现过**(可能只在问题里出现过 ⇒ emb 受训但 head 未受训)
         ⇒ 候选: **词项问题**(真正的输出头未绑权: head.weight[t] 梯度为零)

⚠ 这比外部建议的"是否在训练任意位置出现过"更严格 —— 关键区分在于**梯度进的是哪张表**。

────────────────────────────────────────────────────────────────────────
**★必须控制的混淆**(§44 已经吃过一次亏)

B/C 组很可能系统性地是"离训练问题更远"的行 ⇒ 组效应可能纯粹是距离效应。
故必须:
  1. 报每组的**平均最近邻距离**与距离分布;
  2. 在**同一距离四分位内**再比 A/B/C —— 只有组内比较才能把距离剥掉。

────────────────────────────────────────────────────────────────────────
**预登记判决规则**(跑之前写死)

  噪声门槛: 组均值差 < 2×标准误(SE = std/√n) ⇒ 记为"不可区分", 不下结论。
  ★"2×SE" 是在看数据前定的, 不许事后调。

  R1  C 组为空或 n<5 ⇒ **词项假设在本数据上不可测**(BPE 只用 train 拟合,
     词表 token 必然都在训练文本里出现过)。此时只剩 A vs B, 直接跳到 R2。
  R2  Δ_B ≈ Δ_A(不可区分) 且 Δ_C 显著更低 ⇒ **词项为主** ⇒ 需绑权/未登录词处理
  R3  Δ_B 显著低于 Δ_A, 且 Δ_C ≤ Δ_B ⇒ **角色为主** ⇒ 训练时需让更多 token
     出现在答案开头位置
  R4  Δ_B 显著低于 Δ_A **且** Δ_C 显著低于 Δ_B ⇒ **两者都要处理**
  R5  Δ_B ≈ Δ_A ≈ Δ_C ⇒ 三层无差别 ⇒ 答案侧不是本问题的来源, 回去查距离/第三来源

**诚实边界**: v11 单 seed, 无种子极差。关联不等于因果(分层只剥共线, 不构成干预)。
  BPE 只在 train split 上拟合 ⇒ "词表外 token"在本设置下**不可能出现**,
  这本身限制了 C 组的存在, 预登记 R1 就是为它准备的。
"""
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BS = 32
R = 20
SE_MULT = 2.0        # 预登记: 差值需超过 2×SE 才算可区分


@torch.no_grad()
def score_rows(model, rows, tok, max_len, want_pred=False):
    from dialogue_small import encode_examples
    x, starts, lengths = encode_examples(rows, tok, max_len)
    f1, preds = [], []
    for i in range(0, x.shape[0], BS):
        xb = x[i:i + BS].to(DEV)
        sb = starts[i:i + BS].to(DEV)
        lg = model(xb).float()
        bi = torch.arange(xb.shape[0], device=DEV)
        lp = F.log_softmax(lg, dim=-1)
        f1.append(lp[bi, sb - 1].gather(1, xb[bi, sb].unsqueeze(1)).squeeze(1).cpu())
        if want_pred:
            preds.append(lg[bi, sb - 1].argmax(-1).cpu())
    return torch.cat(f1), (torch.cat(preds) if want_pred else None)


def main():
    from tokenizers import Tokenizer
    from model import RCfg, EngramRuleLM
    from dialogue_small import split_records, encode_examples, format_prefix, format_record

    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    seq = int(ck["training"]["seq"])
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model = model.eval()

    tr_rows, ev_rows, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                        seed=29, return_meta=True)
    ev_x, _, _ = encode_examples(ev_rows, tok, seq)
    assert bool((ev_x == ck["eval_tokens"]).all()), "S1 重建不一致"
    print("S1 eval 重建逐元素相同 ✓")

    # ---- 训练答案里的 token 出现计数(任意位置 / 首位置) ----
    ans_any, ans_first = Counter(), Counter()
    q_any = Counter()
    for r in tr_rows:
        pf = tok.encode(format_prefix(r), add_special_tokens=False).ids
        full = tok.encode(format_record(r), add_special_tokens=False).ids
        ans_ids = full[len(pf):]
        ans_any.update(ans_ids)
        if ans_ids:
            ans_first[ans_ids[0]] += 1
        q_any.update(pf)
    print("训练答案 token 种类 = %d / 词表 %d ; 答案首 token 种类 = %d"
          % (len(ans_any), ck["cfg"]["vocab"], len(ans_first)))

    # ---- 可比子集 ----
    plens = [len(tok.encode(format_prefix(r), add_special_tokens=False).ids) for r in ev_rows]
    rlens = [len(tok.encode(format_record(r), add_special_tokens=False).ids) for r in ev_rows]
    alens = [max(0, rlens[i] - plens[i] - 1) for i in range(len(ev_rows))]
    mx = max(alens)
    keep = [i for i in range(len(ev_rows)) if plens[i] + mx + 1 <= seq]
    rows = [ev_rows[i] for i in keep]
    n = len(rows)
    qs = [r["question"] for r in rows]
    ans = [r["answer"] for r in rows]
    firsts = []
    for r in rows:
        pf = tok.encode(format_prefix(r), add_special_tokens=False).ids
        full = tok.encode(format_record(r), add_special_tokens=False).ids
        firsts.append(full[len(pf)] if len(full) > len(pf) else 0)
    firsts = torch.tensor(firsts)
    print("可比子集 n=%d\n" % n)

    # ---- 分组 ----
    grp = torch.tensor([2 if ans_any[t] == 0 else (0 if ans_first[t] > 0 else 1)
                        for t in firsts.tolist()])
    names = {0: "A 当过开头", 1: "B 出现过未当开头", 2: "C 答案里从未出现"}

    # ---- 逐行 Δ_i ----
    own, pred = score_rows(model, [{"question": q, "answer": a, "history": []}
                                   for q, a in zip(qs, ans)], tok, seq, want_pred=True)
    acc = torch.zeros(n)
    g = torch.Generator().manual_seed(29)
    for _ in range(R):
        perm = torch.randperm(n, generator=g).tolist()
        pf, _ = score_rows(model, [{"question": qs[i], "answer": ans[perm[i]], "history": []}
                                   for i in range(n)], tok, seq)
        acc += pf
    di = own - acc / R

    # ---- 最近邻距离(作混淆检查) ----
    tr_qs = [set(tok.encode(format_prefix(r), add_special_tokens=False).ids) for r in tr_rows]
    dist = []
    for q in qs:
        s = set(tok.encode(format_prefix({"question": q, "answer": "", "history": []}),
                           add_special_tokens=False).ids)
        dist.append(1.0 - max(len(s & t) / len(s | t) for t in tr_qs))
    dist = torch.tensor(dist)

    print("逐行 Δ_i: mean=%+.4f std=%.4f\n" % (di.mean(), di.std()))
    print("=== 三组总览 ===")
    print("  %-18s %5s %10s %8s %10s" % ("组", "n", "Δ_i 均值", "SE", "平均距离"))
    stats = {}
    for k in (0, 1, 2):
        m = grp == k
        nn = int(m.sum())
        if nn == 0:
            print("  %-18s %5d %10s %8s %10s" % (names[k], 0, "—", "—", "—"))
            continue
        mu = float(di[m].mean())
        se = float(di[m].std()) / (nn ** 0.5) if nn > 1 else float("nan")
        stats[k] = (nn, mu, se)
        print("  %-18s %5d %10.4f %8.4f %10.3f" % (names[k], nn, mu, se, float(dist[m].mean())))

    # ---- 混淆检查: B/C 是否系统性更远 ----
    print("\n=== 混淆检查: 各组的距离分布(是否 B/C 系统性更远) ===")
    for k in (0, 1, 2):
        m = grp == k
        if int(m.sum()) == 0:
            continue
        q = torch.quantile(dist[m].float(), torch.tensor([0.25, 0.5, 0.75]))
        print("  %-18s 距离四分位 = %s" % (names[k], " ".join("%.3f" % v for v in q)))

    # ---- 距离分层内比较 ----
    db = torch.bucketize(dist, torch.quantile(dist, torch.tensor([0.25, 0.5, 0.75])))
    print("\n=== 同一距离层内比较(剥掉距离效应) ===")
    print("  %-18s %s" % ("组\\距离层", "  ".join("d%d" % j for j in range(4))))
    for k in (0, 1, 2):
        cells = []
        for j in range(4):
            m = (grp == k) & (db == j)
            cells.append("n=%-3d Δ=%+.2f" % (int(m.sum()), float(di[m].mean()))
                         if int(m.sum()) > 0 else "n=0        ")
        print("  %-18s %s" % (names[k], "  ".join(cells)))

    # ---- B/C 组模型实际预测了什么 ----
    print("\n=== 低频组模型实际预测的首 token (验证'系统性答反') ===")
    inv = {v: k for k, v in json_vocab().items()}
    for k in (1, 2):
        m = grp == k
        if int(m.sum()) == 0:
            continue
        pc = Counter(pred[m].tolist())
        truth = Counter(firsts[m].tolist())
        print("  %s n=%d" % (names[k], int(m.sum())))
        print("     真值首 token  Top3: %s" % [(inv.get(t, "?"), c) for t, c in truth.most_common(3)])
        print("     模型预测 Top3: %s" % [(inv.get(t, "?"), c) for t, c in pc.most_common(3)])
        hit = float((pred[m] == firsts[m]).float().mean())
        print("     模型首 token 命中率 = %.3f (随机基线 1/%d = %.4f)"
              % (hit, ck["cfg"]["vocab"], 1.0 / ck["cfg"]["vocab"]))

    # ---- 判决 ----
    print("\n[判决对照预登记规则]")
    if 2 not in stats or stats[2][0] < 5:
        print("  ★命中 R1: C 组 n=%s ⇒ **词项假设在本数据上不可测**"
              % (stats[2][0] if 2 in stats else 0))
    if 0 in stats and 1 in stats:
        nA, mA, sA = stats[0]
        nB, mB, sB = stats[1]
        se = (sA ** 2 + sB ** 2) ** 0.5
        diffB = mA - mB
        print("  Δ_A - Δ_B = %+.4f  (2×SE=%.4f) → %s"
              % (diffB, SE_MULT * se, "可区分" if abs(diffB) > SE_MULT * se else "不可区分"))
    return 0


def json_vocab():
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    import json
    return json.loads(ck["tokenizer"]["json"])["model"]["vocab"]


if __name__ == "__main__":
    sys.exit(main())