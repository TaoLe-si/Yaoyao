"""eval 侧到底差到什么程度?——先量**无脑基线**, 再谈 Transformer 泛化。

桃的怀疑: "Transformer 在玩具模型上泛化表现也很差"。
本脚本回答的是**可证伪的那一半**: 当前读数(eval token acc / eval 首token acc)
究竟比"什么都不学"高多少?

**为什么不猜**: §39 已证明 79% 的 token 准确率来自答案语言模式(错配问题后
全答案 NLL 只 +0.81), 所以"准确率高"本来就不能证明泛化。要判"泛化差",
必须先有 floor。

**基线全部只用 train 统计**, 绝不看 eval 标签以外的任何东西;
eval 只用来"打分"。所有数字都在同一 eval_tokens / eval_answer_starts 上算
(与 §48 的口径一致, 可直接对比)。
"""
import os
import sys
from collections import Counter, defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
SEQ = 512


def main():
    from tokenizers import Tokenizer
    from dialogue_small import split_records, encode_examples

    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr_rows, ev_rows, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                        seed=29, return_meta=True)
    tr_x, tr_s, tr_l = encode_examples(tr_rows, tok, SEQ)
    ev_x, ev_s, ev_l = encode_examples(ev_rows, tok, SEQ)
    assert bool((ev_x == ck["eval_tokens"]).all()), "eval 口径与 v11 不一致"
    assert ev_s.tolist() == ck["extra"]["eval_answer_starts"], "eval starts 不一致"

    # ---- 收集 train / eval 的答案 token（严格按 answer_start 切） ----
    def answers(x, starts, lengths):
        out = []
        for i in range(x.shape[0]):
            s, L = int(starts[i]), int(lengths[i])
            out.append(x[i, s:L].tolist())
        return out

    tr_a = answers(tr_x, tr_s, tr_l)
    ev_a = answers(ev_x, ev_s, ev_l)

    # ---- B0 规模 ----
    tr_first = Counter(a[0] for a in tr_a if a)
    ev_first = Counter(a[0] for a in ev_a if a)
    tr_types = Counter(t for a in tr_a for t in a)
    ev_types = Counter(t for a in ev_a for t in a)
    print("train: %d 条, 答案 token %d 个, 种类 %d" % (len(tr_a), sum(len(a) for a in tr_a),
                                                       len(tr_types)))
    print("eval : %d 条, 答案 token %d 个, 种类 %d" % (len(ev_a), sum(len(a) for a in ev_a),
                                                       len(ev_types)))
    print("eval 答案首 token: %d 种, 最多的一种占 %d/%d = %.1f%%"
          % (len(ev_first), ev_first.most_common(1)[0][1], len(ev_a),
             100.0 * ev_first.most_common(1)[0][1] / len(ev_a)))
    unseen_ev = sum(v for k, v in ev_types.items() if k not in tr_types)
    print("eval 里 train 从未见过的 token: %d 个 (%.1f%%)"
          % (unseen_ev, 100.0 * unseen_ev / max(1, sum(ev_types.values()))))

    def score(pred_fn, name):
        hit = tot = 0
        for a in ev_a:
            for j, t in enumerate(a):
                hit += int(pred_fn(j, t) == t)
                tot += 1
        return name, hit / max(1, tot), tot

    # ---- B1 全局 unigram（最高频 token，闭式 argmax，闭式 = 无脑） ----
    top1 = tr_types.most_common(1)[0][0]
    r = [score(lambda j, t, p=top1: p, "")]
    # 期望命中率（按 train 分布随机抽一个）——这才是"无脑"的公平版
    exp_uni = sum(c * (c / sum(tr_types.values())) for c in tr_types.values()) / sum(tr_types.values())
    print("\n=== eval token 级基线 ===")
    print("  B1 全局最高频单个 token (%r)      acc = %.4f" % (tok.id_to_token(top1), r[0][1]))
    print("     按 train 频率**随机**抽一个的期望 acc = %.4f" % exp_uni)

    # ---- B2 位置条件 unigram（第 j 个答案 token 处最常出现的 token） ----
    bypos = defaultdict(Counter)
    for a in tr_a:
        for j, t in enumerate(a):
            bypos[j][t] += 1
    def b2(j, t):
        c = bypos.get(j)
        return c.most_common(1)[0][0] if c else -1
    hit = tot = 0
    for a in ev_a:
        for j, t in enumerate(a):
            hit += int(b2(j, t) == t); tot += 1
    print("  B2 位置条件最高频 token              acc = %.4f" % (hit / max(1, tot)))

    # ---- B3 答案内 bigram（上一个 token 决定下一个） ----
    bg = defaultdict(Counter)
    for a in tr_a:
        for j in range(1, len(a)):
            bg[a[j - 1]][a[j]] += 1
    def b3(j, t, cur=None):
        return -1
    hit = tot = 0
    for a in ev_a:
        for j, t in enumerate(a):
            pred = a[0] if j == 0 else (bg[a[j - 1]].most_common(1)[0][0]
                                        if bg.get(a[j - 1]) else top1)
            hit += int(pred == t); tot += 1
    print("  B3 train 答案内 bigram               acc = %.4f" % (hit / max(1, tot)))

    # ---- B4 eval 首 token 基线 ----
    print("\n=== eval 首 token 基线 ===")
    print("  B4 恒定输出 train 最高频开头 (%r)   acc = %.4f"
          % (tok.id_to_token(tr_first.most_common(1)[0][0]),
             ev_first.get(tr_first.most_common(1)[0][0], 0) / len(ev_a)))
    print("     按 train 开头频率**随机**抽的期望 = %.4f"
          % (sum(c * (c / sum(tr_first.values())) for c in tr_first.values())
             / sum(tr_first.values())))
    print("     完全不看问题、只看 eval 开头分布的众数上限 = %.4f"
          % (ev_first.most_common(1)[0][1] / len(ev_a)))
    print("\n  对照: v11(d=64) eval 首token acc=0.1078 token acc=0.2560")
    return 0


if __name__ == "__main__":
    sys.exit(main())