"""★预登记 Δ 诊断: 测"问题条件信号"到底存不存在。

这是唯一能把两个互斥解释分开的岔路口:
  (a) 欠拟合/容量不足 —— 有信号, 只是没学好
  (b) 无可迁移监督信号 —— 结构上就没有
若不测这一项, 任何"改架构"或"加训练"的建议都只是猜。

────────────────────────────────────────────────────────────────────────
**定义**

  S_K(q, a) = (1/K) Σ_{t=0}^{K-1} log P(a_t | q, a_{<t})   —— 只在**答案的前 K 个 token**
  Δ(π)      = mean_i [ S_K(qᵢ, aᵢ) − S_K(qᵢ, a_{π(i)}) ]  —— π 为答案的随机置换

主口径 **K=1(只算首 token)**。理由: §38 的错配负控已证明问题条件信号几乎全部集中在首 token
(错配后全答案 NLL 仅 +0.81, 而首 token NLL +6.28), 且 §39/§41 的结论都锚在首 token 上。
K=all 作为次要口径一并报出, 防止"只挑有利口径"。

**Δ 的零假设是精确的**: 若打分与问题无关(S=f(a)), 则 π 是置换 ⇒ {a_{π(i)}} 与 {aᵢ} 同多重集
⇒ Δ(π) ≡ 0。这给了我们两样东西: 一个**精确**的实现自检, 和一个可证伪的判据。

────────────────────────────────────────────────────────────────────────
**三重对照(缺一不可)**

  C1 **恒等问题** —— 把所有问题换成同一句常量, 问题信息归零。
     预登记: Δ 必须 **≈ 0(精确, 容差 1e-6)**。
     若不接近 0 ⇒ **Δ 的实现是错的**, 后面所有读数作废(先读检查本身)。
  C2 **真实正向对照** —— memorize-8 checkpoint 在它**已记住**的 8 条训练题上(§37 三 seed 8/8,
     答案 NLL 0.015)。预登记: Δ 必须**显著为正**。若不为正 ⇒ 判据连已知有效信号都测不出。
     ★用现成 artifact 当对照, 而不是造一个玩具 oracle。
  C3 **零分布** —— 固定种子跑 R 个随机置换, 报 Δ 的 mean/std/min/max。
     "Δ≈0" 必须相对这个分布而言, 否则是绝对阈值口头话。

────────────────────────────────────────────────────────────────────────
**预登记判决规则**(跑之前写死)

  D1 C1 的 |Δ| ≤ 1e-6          ⇒ 实现正确
  D2 C2 的 Δ 显著 > 0           ⇒ 判据在正方向有效
  D3 v11 **train** Δ ≈ 0        ⇒ 连训练题都没有问题条件信号 ⇒ 纯容量/优化问题, 与"缺通路"无关
  D4 v11 train Δ > 0 但 eval Δ ≈ 0 ⇒ 信号存在但**不迁移** ⇒ 真正的泛化缺口

**已知混淆(不消除, 只声明)**: eval 侧答案可能含模型从未学过的 token(输出头不绑权),
  会把 logP 整体压低。Δ 是**差分**, 理论上仍可比, 但 D4 的 eval 侧结论要带这个保留。

**诚实边界**: v11 / memorize-8 均为单 seed, **无种子极差**。
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
MEM8 = os.path.join(HERE, "engram_rule_dialogue_memorize8_budget192_v3.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BS = 32
CONST_Q = "请问"      # C1 的常量问题


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    from model import RCfg, EngramRuleLM
    m = EngramRuleLM(RCfg(**ck["cfg"]))
    m.load_state_dict(ck["state_dict"])
    m = m.to(DEV).eval()
    return ck, m


@torch.no_grad()
def score_rows(model, rows, tok, max_len):
    """返回 (first_lp (N,), all_lp (N,)) —— 每行 [BOS][USER]q[ASSISTANT]a[EOS] 的答案 token 平均 logP。
    行内跳过超长/非法(调用方已预筛, 这里仍做防御)。"""
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
        tgt = xb[:, 1:]
        m = (p >= (sb - 1).view(-1, 1)) & (p < (lb - 1).view(-1, 1)).clamp_min(0)
        tl = lp[:, :-1].gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        s = (tl * m).sum(1)
        fa.append((s / m.sum(1).clamp_min(1)).cpu())
    return torch.cat(f1), torch.cat(fa)


def _prefix_len(tok, row):
    from dialogue_small import format_prefix
    return len(tok.encode(format_prefix(row), add_special_tokens=False).ids)


def fit_subset(rows, tok, max_len):
    from dialogue_small import format_record
    """只保留**所有配对都能放下**的行, 保证任意置换下样本集相同, Δ 才可比。

    ⚠ 2026-10-10 修过的 bug: 原来用 max(整条记录长度) 当"最长答案",
      而整条记录 = 前缀 + 答案 + EOS, 于是 prefix + 整条记录 把前缀算了两遍,
      结果 keep 恒为空(n=0)。现在分开算: 答案长度 = 记录长 − 前缀长 − 1(EOS)。
    """
    plens = [_prefix_len(tok, r) for r in rows]
    rlens = [len(tok.encode(format_record(r), add_special_tokens=False).ids)
             for r in rows]
    alens = [max(0, rlens[i] - plens[i] - 1) for i in range(len(rows))]
    max_ans = max(alens) if alens else 0
    keep = [i for i in range(len(rows)) if plens[i] + max_ans + 1 <= max_len]
    return keep, max_ans


def delta(own_f, own_a, perm_f, perm_a, n):
    idx = torch.arange(n)
    return float(own_f.mean() - perm_f.mean()), float(own_a.mean() - perm_a.mean())


def run_block(name, ck, model, rows, tok, max_len, R, seed):
    from dialogue_small import format_record
    keep, max_ans = fit_subset(rows, tok, max_len)
    n_total = len(rows)
    if not keep:
        print("  [%s] ★无可比子集(n=0), 跳过" % name)
        return None
    rows = [rows[i] for i in keep]
    n = len(rows)
    qs = [r["question"] for r in rows]
    ans = [r["answer"] for r in rows]
    print("  [%s] 可比子集 n=%d/%d (最长答案 %d token, max_len=%d; 丢弃 %d 条以保证"
          "任意置换样本集相同)" % (name, n, n_total, max_ans, max_len, n_total - n))
    t0 = time.time()
    own = [{"question": q, "answer": a, "history": []} for q, a in zip(qs, ans)]
    own_f, own_a = score_rows(model, own, tok, max_len)
    print("      原配对打分完成 %.1fs | 首token 平均 logP = %.4f" % (time.time() - t0, float(own_f.mean())))

    g = torch.Generator().manual_seed(seed)
    d1s, dAs = [], []
    for r in range(R):
        perm = torch.randperm(n, generator=g).tolist()
        cross = [{"question": qs[i], "answer": ans[perm[i]], "history": []} for i in range(n)]
        pf, pa = score_rows(model, cross, tok, max_len)
        d1, dA = delta(own_f, own_a, pf, pa, n)
        d1s.append(d1)
        dAs.append(dA)
    t1s = torch.tensor(d1s)
    tAs = torch.tensor(dAs)
    print("      随机错配 Δ (K=1)  : mean=%+.4f  std=%.4f  min=%+.4f  max=%+.4f  (R=%d)"
          % (t1s.mean(), t1s.std(), t1s.min(), t1s.max(), R))
    print("      随机错配 Δ (K=all): mean=%+.4f  std=%.4f  min=%+.4f  max=%+.4f"
          % (tAs.mean(), tAs.std(), tAs.min(), tAs.max()))

    # C1: 恒等问题 -> Δ 必须精确为 0
    g2 = torch.Generator().manual_seed(seed + 1)
    perm = torch.randperm(n, generator=g2).tolist()
    const_own = [{"question": CONST_Q, "answer": a, "history": []} for a in ans]
    const_cross = [{"question": CONST_Q, "answer": ans[perm[i]], "history": []} for i in range(n)]
    cf, ca = score_rows(model, const_own, tok, max_len)
    xf, xa = score_rows(model, const_cross, tok, max_len)
    c1_1, c1_A = float(cf.mean() - xf.mean()), float(ca.mean() - xa.mean())
    print("      C1 恒等问题 Δ (K=1)=%+.3e  (K=all)=%+.3e  → %s"
          % (c1_1, c1_A, "实现正确" if abs(c1_1) < 1e-6 else "★实现有错, 读数作废"))
    return dict(n=n, d1_mean=float(t1s.mean()), d1_std=float(t1s.std()),
                d1_min=float(t1s.min()), d1_max=float(t1s.max()),
                dA_mean=float(tAs.mean()), dA_std=float(tAs.std()), c1=abs(c1_1))


def main():
    from tokenizers import Tokenizer
    from dialogue_small import split_records, encode_examples
    print("DEV =", DEV)

    # ---- C2 正向对照: memorize-8 已记住的 8 条训练题 ----
    print("\n=== C2 真实正向对照: memorize-8 (三 seed 8/8, 答案 NLL 0.015) ===")
    ck8, m8 = load(MEM8)
    tok8 = Tokenizer.from_str(ck8["tokenizer"]["json"])
    tr8 = ck8["extra"]["experiment"]["prompts"]["train"]
    r2 = run_block("memorize8-train", ck8, m8, tr8, tok8, int(ck8["training"]["seq"]), R=20, seed=101)
    del m8

    # ---- 测量: v11 ----
    print("\n=== v11 测量 ===")
    ck, model = load(V11)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    seq = int(ck["training"]["seq"])
    tr_rows, ev_rows, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                        seed=29, return_meta=True)
    print("  重建 train=%d eval=%d | 训练 seq=%d" % (len(tr_rows), len(ev_rows), seq))
    ev_x, ev_starts, ev_lens = encode_examples(ev_rows, tok, seq)
    assert ev_x.shape == ck["eval_tokens"].shape and bool((ev_x == ck["eval_tokens"]).all()), \
        "S1 重建不一致 (x%s vs ck%s)" % (tuple(ev_x.shape), tuple(ck["eval_tokens"].shape))
    print("  S1 eval 重建逐元素相同 ✓ (x%s)" % (tuple(ev_x.shape),))

    rt = run_block("v11-train", ck, model, tr_rows, tok, seq, R=20, seed=29)
    re_ = run_block("v11-eval", ck, model, ev_rows, tok, seq, R=20, seed=29)

    print("\n[判决对照预登记规则]")
    c1s = [r["c1"] for r in (r2, rt, re_) if r]
    print("  D1 C1 恒等问题 Δ≈0        : %s (max|Δ|=%.2e)"
          % ("通过" if max(c1s) < 1e-6 else "★失败", max(c1s)))
    print("  D2 C2 memorize-8 Δ > 0   : Δ(K=1)=%+.4f  %s"
          % (r2["d1_mean"], "通过" if r2["d1_mean"] > 0 else "★失败"))
    for nm, r in (("v11-train", rt), ("v11-eval", re_)):
        if r is None:
            continue
        sig = r["d1_mean"] > r["d1_std"]
        print("  %-11s Δ(K=1)=%+.4f ± %.4f  |K=all %+.4f ± %.4f | n=%d  → %s"
              % (nm, r["d1_mean"], r["d1_std"], r["dA_mean"], r["dA_std"], r["n"],
                 "显著为正" if sig else "与零分布不可区分"))
    return 0


if __name__ == "__main__":
    sys.exit(main())