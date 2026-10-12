"""测"架构瓶颈": 在决定答案首 token 的位置, 窗口能看到的信息**够不够**区分问题。

桃怀疑是架构问题。本脚本给出**信息上限**的实测, 不猜。

────────────────────────────────────────────────────────────────────────
**问的是什么**

预测首 token 时, 模型在位置 `start−1` 上做决策。它能看到的信息只有:
  · **Engram 检索** —— 已实测: 键 = 问题末 1~2 个 token, 204 条留出问题 95.1% 相同
    ⇒ 这一路**在绑定时刻几乎不携带问题身份**, 上限已定死。
  · **窗口** —— offsets [0,15], 即问题的最后 15 个 token(叠 2 层, 逐层扩大可达范围)

所以关键问题变成: **只用问题的最后 15 个 token, 能区分多少个不同的训练问题?**

若大量问题的末 15 token 相同 ⇒ 架构**信息瓶颈**成立, 换更强的模型也没用。
若几乎全不同 ⇒ 瓶颈不在"看得到多少", 而在"学不学得会" ⇒ 不是架构信息问题。

**顺带**: 因为层数有限, 逐层的可达范围也不同, 故报告 W=8/15/32 三档。
"""
import os
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    from tokenizers import Tokenizer
    from dialogue_small import split_records, format_prefix, format_record

    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr_rows, ev_rows, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                        seed=29, return_meta=True)

    def prefix_ids(r):
        return tok.encode(format_prefix(r), add_special_tokens=False).ids

    tr_p = [prefix_ids(r) for r in tr_rows]
    ev_p = [prefix_ids(r) for r in ev_rows]

    print("训练问题 %d 条, 留出问题 %d 条" % (len(tr_p), len(ev_p)))
    print("完整问题前缀去重: train %d / eval %d"
          % (len({tuple(p) for p in tr_p}), len({tuple(p) for p in ev_p})))

    # ── 口径自检: tail() 剥掉的就是 [ASSISTANT], 不是碰巧等于 5 ────────────
    # 注意: `tokenizers` 的 decode() 默认 skip_special_tokens=True, 特殊 token 解出来是空串
    # (本轮已踩: 断言拿到 {''} 而非 {"[ASSISTANT]"}, 是检查写错不是数据错)。
    # 必须用 id_to_token —— 它不做任何 special 过滤。
    last_ids = {p[-1] for p in tr_p + ev_p}
    last_toks = {tok.id_to_token(i) for i in last_ids}
    assert all(len(p) >= 3 for p in tr_p + ev_p), "前缀过短, 口径不可信"
    assert last_toks == {"[ASSISTANT]"}, \
        "末 token 不是 [ASSISTANT](id=%r tok=%r), tail 口径错" % (last_ids, last_toks)
    print("\n口径自检通过: 全部 %d 条前缀末 token 均为 [ASSISTANT](id=%s), tail() 剥离正确"
          % (len(tr_p) + len(ev_p), sorted(last_ids)))

    def tail(p, w):
        # format_prefix 以 [BOS][USER]q[ASSISTANT] 结尾; [ASSISTANT] 不携带问题信息
        body = p[:-1]
        return tuple(body[-w:])

    def report(label, tails_tr, tails_ev):
        c = Counter(tails_tr)
        uniq = len(c)
        maxgrp = c.most_common(1)[0][1]
        collide = sum(v for v in c.values() if v > 1)
        tr_set = set(tails_tr)
        hit = sum(1 for t in tails_ev if t in tr_set)
        print("\n%-22s 训练去重 %d 组 / %d 条 (碰撞 %d 条, 最大组 %d)"
              % (label, uniq, len(tails_tr), collide, maxgrp))
        print("%-22s 留出撞上训练的: %d / %d (%.1f%%)"
              % ("", hit, len(tails_ev), 100.0 * hit / len(tails_ev)))

    # ── 主测量: 窗口在 start−1 能看到的最后 W 个 token ─────────────────────
    for W in (4, 8, 15, 32):
        report("W=%d (窗口)" % W, [tail(p, W) for p in tr_p], [tail(p, W) for p in ev_p])

    # ── 负向对照(规则 7): 同一个量, 换成 Engram 自己的检索键 ────────────────
    # Engram ngram_max=2 ⇒ 检索键就是末 1 / 末 2 个 token(§39 实测退化)。
    # 若本量在反例下确实崩塌, 才有资格说 W=15 的低碰撞是"真干净"而非恒真。
    for W in (1, 2):
        report("NC: Engram 键 W=%d" % W, [tail(p, W) for p in tr_p], [tail(p, W) for p in ev_p])

    print("\n[读法]")
    print("  · 若『碰撞』接近 0 且『留出命中』接近 0%(即留出问题的末 W token 训练里都没有),")
    print("    说明窗口看到的部分**不构成信息瓶颈** —— 看不到 ≠ 混淆。")
    print("  · 若碰撞很大, 才是架构级的信息瓶颈(不同问题在窗口里长一样)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())