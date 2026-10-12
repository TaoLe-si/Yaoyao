"""★预登记: 拆解"泛化只保留 21%"到底出在问题侧还是答案侧。

§42 实测: v11 的问题条件信号 Δ(train)=+5.17, Δ(eval)=+1.10 ⇒ **只迁移了 21%**。
本脚本回答: 这 21% 的损失是 **问题侧泛化不足** 还是 **答案侧受限**。

────────────────────────────────────────────────────────────────────────
**★先拒绝一个方法: 绑权消融(把 head 权重换成 emb^T)不可用**

外部建议提出"临时把 head 绑到 emb, 看 eval Δ 是否上升"。**这个实验无效**, 理由:

  已训练模型的 `head.weight` 与 `emb.weight` 是两份完全不同的张量。强行绑定会
  **摧毁整个输出分布** —— Δ 的两项(原配对/错配)同时变成垃圾。
  而**被摧毁的模型给出的 Δ 也 ≈ 0**, 与"答案侧受限 ⇒ Δ 低"**不可区分**。
  ⇒ 它把"答案侧是主因"和"我把模型搞坏了"混成同一个读数, 是**假阴性陷阱**。

**替代方案(非破坏性, 直接测答案侧)**:
  用**首答案 token 在训练答案里作为首 token 出现过的频次**做分层。
  若 Δ_i 随该频次单调上升 ⇒ 低 Δ 集中在"训练里几乎没出现过的首 token"上 ⇒ 答案侧受限。
  这不需要动任何权重, 且读数可解释。

────────────────────────────────────────────────────────────────────────
**★混淆必须处理**: 问题侧与答案侧**天然相关** —— 问题相似的两条记录, 答案往往也相似,
于是"最近邻距离"与"首 token 频次"共变。**只算单变量相关必然过度归因**。
故用**双向残差法**: 在控制住一方的分层均值后, 看另一方是否仍有解释力。

  · corr(distance, Δ − mean_bin(Δ | freq))     ⇒ **问题侧**的独立贡献
  · corr(freq,     Δ − mean_bin(Δ | distance)) ⇒ **答案侧**的独立贡献

────────────────────────────────────────────────────────────────────────
**预登记判决规则**(跑之前写死)

  P1 残差相关 |r| ≥ 0.20 且同号 ⇒ 该侧有独立贡献
  P2 两侧都过 ⇒ 21% 两侧都有, 须分别处理
  P3 两侧都不过 ⇒ 21% 来自**第三个来源**(既非频次也非近邻距离), 须另找
  ⚠ n=203, |r|≈0.20 约对应 p<0.005; 阈值取 0.20 是预先定的, 不许事后调

**已知局限(声明, 不消除)**:
  - "答案侧频次"只是答案侧受限的**代理**, 不等于输出头不绑权本身。
  - 最近邻用 token 集合 Jaccard, 不是语义相似度。
  - v11 单 seed, 无种子极差; 关联不等于因果(分层残差只去掉共线, 不构成干预)。
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BS = 32
R = 20            # 置换轮数, 与 §42 一致
THRESH = 0.20     # 预登记阈值, 不得事后调整


@torch.no_grad()
def score_rows(model, rows, tok, max_len):
    from dialogue_small import encode_examples
    x, starts, lengths = encode_examples(rows, tok, max_len)
    f1, fa = [], []
    for i in range(0, x.shape[0], BS):
        xb = x[i:i + BS].to(DEV)
        sb = starts[i:i + BS].to(DEV)
        lb = lengths[i:i + BS].to(DEV)
        lg = model(xb).float()
        bi = torch.arange(xb.shape[0], device=DEV)
        lp = F.log_softmax(lg, dim=-1)
        f1.append(lp[bi, sb - 1].gather(1, xb[bi, sb].unsqueeze(1)).squeeze(1).cpu())
        p = torch.arange(max_len - 1, device=DEV).view(1, -1)
        m = (p >= (sb - 1).view(-1, 1)) & (p < (lb - 1).view(-1, 1)).clamp_min(0)
        tl = lp[:, :-1].gather(2, xb[:, 1:].unsqueeze(-1)).squeeze(-1)
        fa.append(((tl * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
    return torch.cat(f1), torch.cat(fa)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return float((a @ b) / d) if d > 0 else float("nan")


def resid_by_bin(y, xbin, nb):
    """减去按 xbin 分箱后的 y 均值 => 去掉该分层解释掉的部分。"""
    out = y.clone()
    for k in range(nb):
        m = xbin == k
        if int(m.sum()) > 0:
            out[m] = out[m] - out[m].mean()
    return out


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
    print("S1 eval 重建逐元素相同 ✓ | train=%d eval=%d" % (len(tr_rows), len(ev_rows)))

    # 可比子集: 任意配对都放得下 (与 §42 同一处理)
    plens = [len(tok.encode(format_prefix(r), add_special_tokens=False).ids) for r in ev_rows]
    rlens = [len(tok.encode(format_record(r), add_special_tokens=False).ids) for r in ev_rows]
    alens = [max(0, rlens[i] - plens[i] - 1) for i in range(len(ev_rows))]
    max_ans = max(alens)
    keep = [i for i in range(len(ev_rows)) if plens[i] + max_ans + 1 <= seq]
    rows = [ev_rows[i] for i in keep]
    n = len(rows)
    qs = [r["question"] for r in rows]
    ans = [r["answer"] for r in rows]
    print("可比子集 n=%d/%d\n" % (n, len(ev_rows)))

    # ---- 逐行 Δ_i ----
    own_f, _ = score_rows(model, [{"question": q, "answer": a, "history": []}
                                  for q, a in zip(qs, ans)], tok, seq)
    acc_perm = torch.zeros(n)
    g = torch.Generator().manual_seed(29)
    for r in range(R):
        perm = torch.randperm(n, generator=g).tolist()
        pf, _ = score_rows(model, [{"question": qs[i], "answer": ans[perm[i]], "history": []}
                                   for i in range(n)], tok, seq)
        acc_perm += pf
    acc_perm /= R
    di = own_f - acc_perm
    print("逐行 Δ_i: mean=%+.4f std=%.4f | 原配对 logP mean=%.4f\n"
          % (di.mean(), di.std(), own_f.mean()))

    # ---- 答案侧协变量: 首答案 token 在训练答案里作为**首 token**出现的频次 ----
    first_tok_train = {}
    for r in tr_rows:
        e = tok.encode(format_record(r), add_special_tokens=False).ids
        pf = tok.encode(format_prefix(r), add_special_tokens=False).ids
        if len(e) > len(pf):
            first_tok_train[e[len(pf)]] = first_tok_train.get(e[len(pf)], 0) + 1
    freq = torch.tensor([float(first_tok_train.get(
        tok.encode(format_record(r), add_special_tokens=False).ids[
            len(tok.encode(format_prefix(r), add_special_tokens=False).ids)], 0))
        for r in rows])
    print("首答案 token 训练频次: 0 次的占 %.1f%%  | 频次分布 中位=%.0f max=%.0f"
          % (100 * float((freq == 0).float().mean()), float(freq.median()), float(freq.max())))

    # ---- 问题侧协变量: 到训练集最近邻的 token 集合 Jaccard 距离 ----
    tr_qsets = [set(tok.encode(format_prefix(r), add_special_tokens=False).ids) for r in tr_rows]
    dist = []
    for q in qs:
        qs_set = set(tok.encode(format_prefix({"question": q, "answer": "",
                                               "history": []}), add_special_tokens=False).ids)
        best = 0.0
        for t in tr_qsets:
            u = len(qs_set | t)
            if u:
                best = max(best, len(qs_set & t) / u)
        dist.append(1.0 - best)
    dist = torch.tensor(dist)
    print("最近邻距离 (1−Jaccard): 中位=%.3f  min=%.3f max=%.3f\n"
          % (float(dist.median()), float(dist.min()), float(dist.max())))

    # ---- 单变量 + 双向残差 ----
    lf = torch.log1p(freq)
    fb = torch.bucketize(freq, torch.tensor([0.5, 3.5, 10.5]))
    db = torch.bucketize(dist, torch.quantile(dist, torch.tensor([0.25, 0.5, 0.75])))
    nfb, ndb = int(fb.max()) + 1, int(db.max()) + 1

    print("单变量相关 (未控制):")
    print("   corr(答案侧 log频次, Δ_i) = %+.3f" % corr(lf, di))
    print("   corr(问题侧 最近邻距离, Δ_i) = %+.3f" % corr(dist, di))
    print("\n双向残差 (控制住另一侧的分层均值):")
    rq = corr(dist, resid_by_bin(di, fb, nfb))
    ra = corr(lf, resid_by_bin(di, db, ndb))
    print("   ★问题侧: corr(距离, Δ−mean(Δ|频次)) = %+.3f   %s" % (rq, "过阈值" if abs(rq) >= THRESH else "未过"))
    print("   ★答案侧: corr(频次, Δ−mean(Δ|距离)) = %+.3f   %s" % (ra, "过阈值" if abs(ra) >= THRESH else "未过"))

    print("\n交叉分层表 (行=频次层, 列=距离层, 格内 Δ_i 均值):")
    print("  %-10s %s" % ("freq\\dist", "  ".join("d%d(n=%d)" % (j, int((db == j).sum()))
                                                     for j in range(ndb))))
    for i in range(nfb):
        cells = []
        for j in range(ndb):
            m = (fb == i) & (db == j)
            cells.append("  n=%-3d Δ=%+.2f" % (int(m.sum()), float(di[m].mean()))
                         if int(m.sum()) > 0 else "  n=0        ")
        print("  f%d(n=%-3d)" % (i, int((fb == i).sum())), "".join(cells))

    print("\n[判决对照预登记规则]")
    p1 = abs(rq) >= THRESH
    p2 = abs(ra) >= THRESH
    if p1 and p2:
        print("  ★命中 P2: 两侧都有独立贡献 ⇒ 21%% 须分别处理")
    elif p1:
        print("  ★命中 P1: 仅**问题侧**有独立贡献 ⇒ 补问题侧泛化通路有意义")
    elif p2:
        print("  ★命中 P1: 仅**答案侧**有独立贡献 ⇒ 先修答案侧(首 token 可输出性)")
    else:
        print("  ○命中 P3: 两侧都不过 ⇒ 21%% 来自第三个来源, 须另找")
    return 0


if __name__ == "__main__":
    sys.exit(main())