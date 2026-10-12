# -*- coding: utf-8 -*-
"""§69.46 train 与 held 之间到底有没有**可迁移的模式**。

## 为什么测这个

held EM 在四个配置上全部为 0（N=2,952/d=256、N=124,593/d=128、N=124,593/d=256、
N=738/d=128），且 train/held 按词项级完全不相交（§69.33 G3）。
若两侧**没有任何共享模式**，则模型只能死记、无从泛化 ⇒ 任何训练技巧都无效。

## 可证伪判据

| 编号 | 量什么 | 有迁移 | 无迁移 |
|---|---|---|---|
| **T1** | held 的问题骨架是否在 train 骨架集合里出现过 | 出现过 ⇒ 句式可迁移 | 完全没出现 ⇒ 零迁移 |
| **T2** | held 的答案骨架是否在 train 答案骨架集合里 | 同上 | 同上 |
| **T3** | 词项级重叠率（字符 bigram 集合的 Jaccard） | >0 ⇒ 有共享词项 | ≈0 ⇒ 完全不相交 |

★骨架口径同 `probe_data_saturation.py`：数字/书名号/英文/机构名替换为 `§`。
★判据必须可失败：若 T1/T2/T3 都 ≈0，则「换训练技巧」这条路整体关闭。

用法：`python probe_transfer.py`（只读语料，不训练）
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_scaling_grid_v2 as E2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "transfer.json")

SLOTS = [
    (re.compile(r"\d+(?:\.\d+)?"), "§"),
    (re.compile(r"《[^》]*》"), "§"),
    (re.compile(r"[A-Za-z][A-Za-z0-9.\-]*"), "§"),
    (re.compile(r"[一-鿿]{2,4}(?:公司|集团|品牌|大学|学院|银行|医院)"), "§"),
]


def skel(t):
    s = t
    for p, r in SLOTS:
        s = p.sub(r, s)
    return s


def bigrams(t):
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def main():
    print("=" * 78)
    print("§69.46 train 与 held 的可迁移模式（只读语料）")
    print("=" * 78)
    E2.load_all()
    rows = E2.load_rows()
    tr, hd = E2.split_rows(rows)
    print("  train %d 行 / held %d 行" % (len(tr), len(hd)))

    # ── T1 问题骨架 ────────────────────────────────────────────────
    sk_tr = {skel(r["question"]) for r in tr}
    sk_hd = [skel(r["question"]) for r in hd]
    hit_q = sum(1 for s in sk_hd if s in sk_tr)
    print()
    print("【T1】held 的问题骨架是否在 train 出现过")
    print("  train 骨架 %d 个；held %d 条中命中 %d（%.2f%%）"
          % (len(sk_tr), len(sk_hd), hit_q, 100.0 * hit_q / len(sk_hd)))

    # ── T2 答案骨架 ────────────────────────────────────────────────
    sa_tr = {skel(r["answer"]) for r in tr}
    sa_hd = [skel(r["answer"]) for r in hd]
    hit_a = sum(1 for s in sa_hd if s in sa_tr)
    print()
    print("【T2】held 的答案骨架是否在 train 出现过")
    print("  train 答案骨架 %d 个；held %d 条中命中 %d（%.2f%%）"
          % (len(sa_tr), len(sa_hd), hit_a, 100.0 * hit_a / len(sa_hd)))

    # ── T3 词项级 bigram 重叠（抽样算，全量太慢）──────────────────
    print()
    print("【T3】字符 bigram 集合的 Jaccard（抽 2000 行算）")
    g = E2.torch.Generator().manual_seed(12345)
    n_s = min(2000, len(tr))
    ti = E2.torch.randperm(len(tr), generator=g)[:n_s].tolist()
    hi = E2.torch.randperm(len(hd), generator=g)[:n_s].tolist()
    # ★用集合交并的近似：合并两侧 bigram 到两个总集，算交集大小
    A, B = set(), set()
    for i in ti:
        A |= bigrams(tr[i]["question"] + tr[i]["answer"])
    for i in hi:
        B |= bigrams(hd[i]["question"] + hd[i]["answer"])
    inter = len(A & B)
    union = len(A | B)
    j = inter / max(union, 1)
    print("  train bigram %d 个 / held bigram %d 个" % (len(A), len(B)))
    print("  交集 %d / 并集 %d ⇒ Jaccard = %.4f" % (inter, union, j))

    # ── 判决 ────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("【判决】")
    print("=" * 78)
    t1r = hit_q / len(sk_hd)
    t2r = hit_a / len(sa_hd)
    print("  T1 问题骨架命中率 = %.4f" % t1r)
    print("  T2 答案骨架命中率 = %.4f" % t2r)
    print("  T3 bigram Jaccard = %.4f" % j)
    if t1r > 0.05 or t2r > 0.05 or j > 0.05:
        print("  ⇒ **存在可迁移模式** ⇒ held EM 恒 0 另有原因（不是语料零信息量）。")
    else:
        print("  ⇒ 三项全部 ≈ 0 ⇒ **train 与 held 之间几乎没有可迁移的模式**。")
        print("     held EM 恒 0 是**语料结构决定的**：问题与答案都不重叠，")
        print("     模型只能死记，没有任何可泛化的规律可学。")
        print("     ⇒ 任何训练技巧（加 d / 加步数 / 正则 / 解码策略）都无效。")
        print("     ⇒ 要 held EM 非 0，必须**改语料切分或内容**。")

    facts = dict(train_rows=len(tr), held_rows=len(hd),
                 T1_q_skel_hit=hit_q, T1_rate=round(t1r, 5),
                 T2_a_skel_hit=hit_a, T2_rate=round(t2r, 5),
                 T3_bigram_jaccard=round(j, 5),
                 train_skel=len(sk_tr), train_ans_skel=len(sa_tr))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
