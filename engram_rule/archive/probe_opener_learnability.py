"""★预登记: 决定"扩充答案开头覆盖"这条修法**是否值得做**。

§45 诊断: eval 答案首 token 没在训练里当过开头的那 13.8%, Δ 全负(系统性答反);
训练里只有 **269** 个 token 被训练成过答案开头。外部据此建议"扩充数据让覆盖更广"。

**但这个建议有一个未验证的前提**:
  模型**能不能学会已经出现过的冷门开头**?
  · 若**能**(train 侧冷门开头也学得好) ⇒ eval 的失败是"没见过"⇒ **扩数据是对的方向**。
  · 若**不能**(train 侧冷门开头也学不好) ⇒ 模型连见过的东西都学不会 ⇒ 这是
    **优化/欠拟合**问题, 扩数据只是多给几遍同样的难例, **收益会很小**。

**这个区分不需要重训就能测** —— 训练集里必然存在冷门开头(820 行只用 269 种开头,
平均每种约 3 次), 在 train 侧分层测 Δ 即可。

────────────────────────────────────────────────────────────────────────
**口径: 留一频次(LOO, leave-one-out)**

对训练行 i, 其答案首 token t_i 的 LOO 频次 =
  `#{ j ≠ i : 训练行 j 的答案首 token == t_i }`
即"**除了它自己**, 训练里还有多少行用同一个开头"。
必须用 LOO 而不是原始频次 —— 否则每个频次=1 的 token 自己贡献 1, 分不清是
"模型没学会"还是"数据里只有一次"。

────────────────────────────────────────────────────────────────────────
**预登记判决规则**(跑之前写死)

  P1  train 侧 Δ 随 LOO 频次**单调下降**, 且 LOO=0 组显著低于高频组
      ⇒ 模型**学不会已出现的冷门开头** ⇒ 扩数据收益有限, 应先查优化侧
  P2  train 侧 Δ 在各频次组**基本持平**(无可区分差异)
      ⇒ 模型**学会了所有出现过的开头** ⇒ eval 失败确实是"没见过"
      ⇒ **扩数据/增广是对的方向**, 值得预登记重训
  P3  LOO=0 组 Δ 为**负** ⇒ 连只出现一次的训练开头都没学会 ⇒ 强 P1 信号

  噪声门槛: 组间差需 > 2×SE(合并), 阈值看数据前定死。

**诚实边界**: v11 单 seed, 无种子极差。train 侧的 Δ 高是**记忆**的体现,
  本实验只问"冷门开头是否被记住", 不回答"泛化"。
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
SE_MULT = 2.0


@torch.no_grad()
def score_rows(model, rows, tok, max_len):
    from dialogue_small import encode_examples
    x, starts, lengths = encode_examples(rows, tok, max_len)
    f1, hit = [], []
    for i in range(0, x.shape[0], BS):
        xb = x[i:i + BS].to(DEV)
        sb = starts[i:i + BS].to(DEV)
        lg = model(xb).float()
        bi = torch.arange(xb.shape[0], device=DEV)
        lp = F.log_softmax(lg, dim=-1)
        f1.append(lp[bi, sb - 1].gather(1, xb[bi, sb].unsqueeze(1)).squeeze(1).cpu())
        hit.append((lg[bi, sb - 1].argmax(-1) == xb[bi, sb]).float().cpu())
    return torch.cat(f1), torch.cat(hit)


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
    assert bool((ev_x == ck["eval_tokens"]).all()), "S1 eval 重建不一致"

    tr_x, _, _ = encode_examples(tr_rows, tok, seq)
    print("S1 eval 重建逐元素相同 ✓ | train=%d eval=%d" % (tr_x.shape[0], ev_x.shape[0]))

    def opener(r):
        pf = tok.encode(format_prefix(r), add_special_tokens=False).ids
        fl = tok.encode(format_record(r), add_special_tokens=False).ids
        return fl[len(pf)] if len(fl) > len(pf) else 0

    top = [opener(r) for r in tr_rows]
    cnt = Counter(top)
    loo = torch.tensor([float(cnt[t] - 1) for t in top])     # ★留一
    print("训练答案开头种类 = %d ; LOO 频次: =0 的 %d 行 (%.1f%%), =1 的 %d 行, 频次中位=%.0f"
          % (len(cnt), int((loo == 0).sum()), 100 * float((loo == 0).float().mean()),
             int((loo == 1).sum()), float(loo.median())))

    # ---- train 侧逐行 Δ ----
    qs = [r["question"] for r in tr_rows]
    ans = [r["answer"] for r in tr_rows]
    plens = [len(tok.encode(format_prefix(r), add_special_tokens=False).ids) for r in tr_rows]
    rlens = [len(tok.encode(format_record(r), add_special_tokens=False).ids) for r in tr_rows]
    alens = [max(0, rlens[i] - plens[i] - 1) for i in range(len(tr_rows))]
    mx = max(alens)
    keep = [i for i in range(len(tr_rows)) if plens[i] + mx + 1 <= seq]
    qk = [qs[i] for i in keep]
    ak = [ans[i] for i in keep]
    loo_k = loo[keep]
    n = len(keep)
    print("可比子集 n=%d/%d\n" % (n, len(tr_rows)))

    own, hit = score_rows(model, [{"question": q, "answer": a, "history": []}
                                  for q, a in zip(qk, ak)], tok, seq)
    acc = torch.zeros(n)
    g = torch.Generator().manual_seed(29)
    for _ in range(R):
        perm = torch.randperm(n, generator=g).tolist()
        pf, _ = score_rows(model, [{"question": qk[i], "answer": ak[perm[i]], "history": []}
                                   for i in range(n)], tok, seq)
        acc += pf
    di = own - acc / R
    print("train 逐行 Δ_i: mean=%+.4f std=%.4f | 首token acc=%.4f\n"
          % (di.mean(), di.std(), float(hit.mean())))

    bins = [(0, 0), (1, 1), (2, 3), (4, 10 ** 9)]
    names = ["LOO=0 (唯一)", "LOO=1", "LOO 2-3", "LOO>=4"]
    print("=== train 侧按留一频次分层 ===")
    print("  %-16s %5s %10s %8s %12s" % ("组", "n", "Δ_i 均值", "SE", "首token acc"))
    stats = []
    for (lo, hi), nm in zip(bins, names):
        m = (loo_k >= lo) & (loo_k <= hi)
        nn = int(m.sum())
        if nn == 0:
            print("  %-16s %5d %10s %8s %12s" % (nm, 0, "—", "—", "—"))
            continue
        mu = float(di[m].mean())
        se = float(di[m].std()) / (nn ** 0.5) if nn > 1 else float("nan")
        stats.append((nm, nn, mu, se, float(hit[m].mean())))
        print("  %-16s %5d %10.4f %8.4f %12.4f" % (nm, nn, mu, se, float(hit[m].mean())))

    print("\n[判决对照预登记规则]")
    if len(stats) >= 2:
        lo0 = stats[0]
        hi_ = stats[-1]
        se = (lo0[3] ** 2 + hi_[3] ** 2) ** 0.5
        d = hi_[2] - lo0[2]
        print("  Δ(LOO>=4) − Δ(LOO=0) = %+.4f  (2×SE=%.4f) → %s"
              % (d, SE_MULT * se, "可区分" if abs(d) > SE_MULT * se else "不可区分"))
        if lo0[2] < 0:
            print("  ★命中 P3: LOO=0 组 Δ 为**负** ⇒ 连只出现一次的训练开头都没学会")
            print("    ⇒ 强 P1 信号: 扩数据只是多给几遍同样的难例, **收益会很小**。")
        elif abs(d) <= SE_MULT * se:
            print("  ★命中 P2: 各频次组 Δ 无可区分差异 ⇒ 模型**学会了所有出现过的开头**")
            print("    ⇒ eval 的失败确实是'没见过' ⇒ **扩数据/增广是对的方向**。")
        else:
            print("  ★命中 P1: Δ 随 LOO 频次下降 ⇒ 模型学不会已出现的冷门开头")
            print("    ⇒ 扩数据收益有限, 应先查优化侧。")
    return 0


if __name__ == "__main__":
    sys.exit(main())