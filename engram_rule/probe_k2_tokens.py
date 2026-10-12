"""§69.14 补充：K2 必须用 **BPE token 数**，不能用 `str.split()`。

**为什么补这一节**：第一版 K2 用 `len(text.split())` 数长度，中文没空格 ⇒
中位数只有 **1 个「词」**，于是「CoT 20x 也装得下 seq=512」——**结论完全反了**。
这是「尺子选错」：中文分词不能按空格。

本脚本用项目自己的 BPE tokenizer 重算，给出权威口径。

用法：`python probe_k2_tokens.py`
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

SEQ = E2.SEQ


def main():
    print("=" * 74)
    print("K2 修正：用 BPE token 数（不是 split()）量长度")
    print("=" * 74)
    rows = E2.load_rows()
    sub = rows[:20000]

    # 错的尺子：split()
    words = sorted(len((r["question"] + r["answer"]).split()) for r in sub)
    n = len(words)
    print()
    print("【错尺子】str.split() 数出来的「词」数（中文无空格，几乎无效）")
    print("  中位 %d / p90 %d / p99 %d / max %d"
          % (words[n // 2], words[int(n * 0.9)], words[int(n * 0.99)], words[-1]))

    # 对的尺子：BPE
    from tokenizers import Tokenizer
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    x, s, l = E2.encode_cached(sub, tok, SEQ)
    al = (l - s).float()             # ★必须转 float：`quantile()` 不接受整数张量
    print()
    print("【对尺子】BPE token 数（answer 段，seq=%d）" % SEQ)
    print("  中位 %d / p90 %d / p99 %d / max %d"
          % (int(al.median()), int(al.quantile(0.9)),
             int(al.quantile(0.99)), int(al.max())))
    print("  ⇒ 被 seq=%d 截断（answer 已满）的行数 = %d / %d"
          % (SEQ, int((al >= SEQ).sum()), n))

    p90 = int(al.quantile(0.9))
    print()
    print("【CoT 倍数推演】（以 p90 = %d 为基线）" % p90)
    for mult, tag in ((1, "纯答案"), (3, "CoT 3x"), (5, "CoT 5x"), (20, "CoT 20x")):
        need = p90 * mult
        print("  %-10s p90 需 ≈ %6d token ⇒ 装得下(%d)? %s"
              % (tag, need, SEQ, "是" if need <= SEQ else "★否"))

    print()
    print("=" * 74)
    print("【修正后的 K2 结论】")
    print("=" * 74)
    print("  · split() 那条**作废**（中文尺子无效，中位 1 个「词」）")
    print("  · 纯答案 p90 = %d token，**已经吃掉 seq=%d 的 %.0f%%" % (p90, SEQ, 100.0 * p90 / SEQ))
    print("  · CoT 只要比答案长 3 倍，p90 就**装不下**")
    print("  · 装一条典型 CoT 样本需要 seq ≈ %d（3x）到 %d（20x）"
          % (p90 * 3, p90 * 20))
    print()
    print("  ⇒ 这与 §69.9 的长上下文障碍**同向**，但量级更紧：")
    print("     不是「要 1M 才装不下」，而是 **3 倍 CoT 就把 512 撑爆**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
