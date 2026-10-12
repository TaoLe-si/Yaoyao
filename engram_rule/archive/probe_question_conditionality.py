"""★判决性测量: 问题条件性**沿答案位置**怎么衰减 —— "答非所问" 的直接机制。

**背景(为什么必须测这个)**: `train_qa_report.md` 的 12 条自由生成样本呈现一个精确模式:
  · #2/#3/#6/#10/#12: 问题问"食物/兴趣/季节/颜色/书", 生成"我最喜欢的**运动**/**篮球**/**秋天**"
    ⇒ **句式模板对了、槽位值填成训练里最高频的那个**
  · #4  反例: 问"电影" → "我最喜欢的**电影**是《肖申克的…", 槽位**对了**
  · #8/#9 反例: "你有**宠物**吗"→"是的,我有一只**猫**"; "你喜欢**做饭**吗"→"是的,我喜欢**尝试新的食谱**"
    ⇒ 问题关键词**紧邻答案开头**时能用上
⇒ 假设: 模型只把问题用在"选开头"(§42 的 Δ 就只测了这一个), **答案内容的位置没用问题**。

**测量**: Δ_k = mean[ logP(a_k | q, a_<k) − logP(a_k | q_π, a_<k) ], π 是别的行的问题。
  · 只换问题、**答案不动** ⇒ a_k 从未出现在前缀里 ⇒ **不构成复制泄漏**(§23 的教训)。
  · q_π 用多组置换取平均, 避免"碰巧与答案同题"的行把 Δ 压低。
  · 这正是 §42 `delta_k1` 的推广(§42 只测 k=1)。

**预登记(可证伪)**:

H1  问题条件性**随答案位置单调衰减**, 且在内容位置趋近 0。
    **否证**: Δ_k 在 k=1..20 上基本持平(或不降) ⇒ H1 被否 ⇒ 问题内容全程被用上了,
    "槽位填错"必须另找原因。

H2  Δ_1 显著大于 Δ_(k>1) 的中位数(>2SE) —— 即"问题只决定开头"确实成立。
    **否证**: 二者 ≤2SE ⇒ H2 被否。

H3  Δ_k 在**所有**位置上都显著 > 0(问题全程被用到)。
    **否证**: 存在某个 k 使 Δ_k 的 2SE 覆盖 0 ⇒ H3 被否。

★负向对照: 恒等问题(q_π = q)时 Δ 必须**精确为 0**(§42 的 C1 已证过, 这里复核)。
  若它不为 0, 说明我这个测量本身有泄漏, 全部读数作废。
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
N_ROWS = 200
N_PERM = 4
MAX_K = 24
BS = 32


def build():
    from tokenizers import Tokenizer
    from model import RCfg, EngramRuleLM
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    m = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return ck, tok, m


@torch.no_grad()
def token_logprobs(model, tok, pairs, max_k=MAX_K):
    """pairs: [(question, answer), ...] → (N, max_k) 每个答案位置的 logP, 越界填 nan。

    口径与 §42 完全一致: 拼 `[BOS][USER]q[ASSISTANT]a`, 位置 starts 处是答案第一个 token。
    """
    from dialogue_small import format_prefix
    texts, starts = [], []
    for q, a in pairs:
        p = tok.encode(format_prefix({"question": q, "history": []}),
                       add_special_tokens=False).ids
        full = tok.encode(format_prefix({"question": q, "history": []}) + a,
                          add_special_tokens=False).ids
        texts.append(full[:SEQ])
        starts.append(len(p))
    n = len(texts)
    out = torch.full((n, max_k), float("nan"))
    for i in range(0, n, BS):
        xs = texts[i:i + BS]
        m = max(len(t) for t in xs)
        xb = torch.full((len(xs), m), 0, dtype=torch.long)
        for j, t in enumerate(xs):
            xb[j, :len(t)] = torch.tensor(t, dtype=torch.long)
        xb = xb.to(DEV)
        lg = model(xb).float()
        lp = F.log_softmax(lg, dim=-1)
        for j, t in enumerate(xs):
            s = starts[i + j]
            for k in range(max_k):
                pos = s + k                     # 预测位置(pos)上的 token 是答案第 k+1 个
                if pos + 1 >= len(t):
                    break
                out[i + j, k] = float(lp[j, pos, t[pos + 1]])
    return out


def curve(delta):
    """delta: (n, K) → 每个 k 的 mean 与 SE(只取该 k 有效的行)。"""
    ms, ses, ns = [], [], []
    for k in range(delta.shape[1]):
        v = delta[:, k]
        v = v[~torch.isnan(v)]
        n = v.numel()
        ns.append(n)
        if n == 0:
            ms.append(float("nan"))
            ses.append(float("nan"))
            continue
        m = float(v.mean())
        ms.append(m)
        ses.append(float(v.std(unbiased=True) / (n ** 0.5)) if n > 1 else float("nan"))
    return ms, ses, ns


def main():
    import exp_capacity_scaling as ecs
    ck, tok, model = build()
    _ck, _tok, tr_rows, ev_rows, tr, ev = ecs.load_data()   # load_data 返回 6 个值
    for name, rows, cap in (("train", tr_rows, N_ROWS), ("eval", ev_rows, N_ROWS)):
        rows = rows[:cap]
        qs = [r["question"] for r in rows]
        ans = [r["answer"] for r in rows]
        base = token_logprobs(model, tok, list(zip(qs, ans)))
        g = torch.Generator().manual_seed(2024)
        acc = torch.zeros_like(base)
        for _ in range(N_PERM):
            perm = torch.randperm(len(rows), generator=g).tolist()
            if all(perm[i] == i for i in range(len(rows))):
                continue                                   # 恒等置换: 跳过, 否则等于白测
            alt = token_logprobs(model, tok, [(qs[perm[i]], ans[i]) for i in range(len(rows))])
            acc += base - alt
        delta = (acc / N_PERM).cpu()
        ms, ses, ns = curve(delta)
        print("\n=== %s: 问题条件性 Δ_k 随答案位置(k=1..%d) ===" % (name, MAX_K))
        print("  k   Δ_k        2SE       有效行")
        for k in range(MAX_K):
            if ns[k] < 5:
                continue
            print("  %-3d %+.4f   %.4f   %d" % (k + 1, ms[k], 2 * ses[k], ns[k]))

        # ---- H1: 单调衰减 ----
        seq = [(ms[k], k) for k in range(MAX_K) if ns[k] >= 5]
        early = sum(ms[k] for k in range(1, min(4, MAX_K)) if ns[k] >= 5) / \
            max(1, sum(1 for k in range(1, min(4, MAX_K)) if ns[k] >= 5))
        late = sum(ms[k] for k in range(5, MAX_K) if ns[k] >= 5) / \
            max(1, sum(1 for k in range(5, MAX_K) if ns[k] >= 5))
        # ---- H2/H3 的显著性 ----
        d1, se1 = ms[0], ses[0]
        tail = [(ms[k], ses[k]) for k in range(1, MAX_K) if ns[k] >= 5]
        tail_med = sorted(t[0] for t in tail)[len(tail) // 2] if tail else float("nan")
        n_zero = sum(1 for v, s in tail if abs(v) <= 2 * s)
        print("  H1 早期(k=1..3)均值 %+.4f vs 后期(k=6..)均值 %+.4f ⇒ %s"
              % (early, late, "衰减成立" if early > late else "★未见衰减"))
        print("  H2 Δ_1=%+.4f (2SE=%.4f) vs Δ_(k>1) 中位 %+.4f ⇒ %s"
              % (d1, 2 * se1, tail_med,
                 "Δ_1 显著更大" if d1 - tail_med > 2 * se1 else "★无显著差别"))
        print("  H3 k>=2 里 2SE 覆盖 0 的位置数 %d/%d ⇒ %s"
              % (n_zero, len(tail), "问题全程被用到" if n_zero == 0 else "★存在位置问题不被用到"))

        # ---- 负向对照: 恒等置换下 Δ 必须精确为 0 ----
        # ★第一版写成 `ident.abs().max()`, 而越界位置是 nan ⇒ nan-nan=nan ⇒ 读数是 nan,
        #   断言炸掉。**不是测量有泄漏, 是检查写错了**(规则 5): 必须只在**有效**位置上比。
        same = token_logprobs(model, tok, list(zip(qs, ans)))
        d_ident = base - same
        fin = d_ident[~torch.isnan(d_ident)]
        mx = float(fin.abs().max()) if fin.numel() else 0.0
        print("  [负向对照] 恒等问题置换: 有效位 max|Δ| = %.3e (要求恰为 0, 有效位 %d)"
              % (mx, fin.numel()))
        assert mx == 0.0, "恒等置换在有效位不为 0 ⇒ 测量有泄漏, 读数作废"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())