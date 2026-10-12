# -*- coding: utf-8 -*-
"""§69.45 检验「数据饱和」假设：训练集的**去重模式数** vs 行数。

## 桃的假设（2026-10-12）

「你确定不是因为数据饱和吗，这种类似的情况我们在小模型上也遇到过。」

## 「数据饱和」的可证伪定义

若成立，则训练集的**去重模式数**远小于行数 —— 模型见过 N 个不同模式后，
剩下几万行都是重复，训练只是在重复已学的东西。加训练量/改架构都无效，
因为**数据里能学的东西已经学尽**。

## 判据（先登记，可失败）

| 编号 | 量什么 | 饱和的方向 | 否定的方向 |
|---|---|---|---|
| **S1** | question 去重数 / 总行数 | 远小于 1（如 <0.1） | 接近 1 |
| **S2** | 槽位抽掉后的**骨架**去重数 | 只有几百/几千 ⇒ 严重饱和 | 上万 ⇒ 句式多样 |
| **S3** | 同 question 的 answer 是否唯一确定 | 若一个 question 只有 1 个 answer ⇒ 记忆任务，饱和更快 | 多个 answer ⇒ 有泛化空间 |

★S2 的做法：把数字、书名号内容、英文品牌等**槽位**替换成占位符 `§`，再去重。
  这能量出「不同的句式骨架有多少个」——这是模型真正要学的模式数。
  AGENTS 规则 13 记载过本项目早期「模板只有 241 种」，所以 S2 是关键判据。

用法：`python probe_data_saturation.py`（只读语料文件，不训练不生成）
"""
import io
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys_path = HERE
import sys
sys.path.insert(0, HERE)

CORPUS = os.path.join(HERE, "corpus_modelscope_v2_155740.jsonl")
OUT = os.path.join(HERE, "data_saturation.json")

# ★槽位正则：数字、书名号、英文串、引号内容。可复现，写死。
SLOTS = [
    (re.compile(r"\d+(?:\.\d+)?"), "§"),        # 数字
    (re.compile(r"《[^》]*》"), "§"),             # 书名号
    (re.compile(r"[A-Za-z][A-Za-z0-9.\-]*"), "§"),   # 英文/品牌
    (re.compile(r"[一-鿿]{2,4}(?:公司|集团|品牌|大学|学院|银行|医院)"), "§"),  # 机构名
]


def skeleton(q):
    """把槽位替换成占位符，得到句式骨架。"""
    s = q
    for pat, rep in SLOTS:
        s = pat.sub(rep, s)
    return s


def main():
    print("=" * 78)
    print("§69.45 数据饱和检验（只读语料，不训练不生成）")
    print("=" * 78)
    if not os.path.exists(CORPUS):
        print("缺 %s" % os.path.basename(CORPUS))
        return 1

    rows = []
    with io.open(CORPUS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "__meta__" in r or not isinstance(r, dict):
                continue
            rows.append(r)
    n = len(rows)
    print("  语料总行数 = %d" % n)

    # ── S1 question 去重 ───────────────────────────────────────────
    qs = [r.get("question", "").strip() for r in rows]
    uq = len(set(qs))
    print()
    print("【S1】question 去重")
    print("  不同 question 数 = %d / %d = %.4f" % (uq, n, uq / n))
    c = Counter(qs)
    top = c.most_common(5)
    print("  最高频 question 出现 %d 次；Top5 频次和 = %d（%.1f%% of 全部）"
          % (top[0][1], sum(v for _, v in top), 100.0 * sum(v for _, v in top) / n))

    # ── S2 骨架去重 ────────────────────────────────────────────────
    sks = [skeleton(q) for q in qs]
    usk = len(set(sks))
    print()
    print("【S2】槽位抽掉后的**骨架**去重")
    print("  不同骨架数 = %d / %d = %.5f" % (usk, n, usk / n))
    cs = Counter(sks)
    top_s = cs.most_common(5)
    print("  Top5 骨架及其频次：")
    for s, v in top_s:
        print("    %5d  %s" % (v, s[:44]))
    print("  ★对照 AGENTS 规则 13 记载的旧合成语料：模板只有 241 种")

    # ── S3 同 question 的 answer 多样性 ────────────────────────────
    q2a = {}
    for r in rows:
        q = r.get("question", "").strip()
        a = r.get("answer", "").strip()
        if q and a:
            q2a.setdefault(q, set()).add(a)
    multi = sum(1 for v in q2a.values() if len(v) > 1)
    print()
    print("【S3】同 question 的 answer 多样性")
    print("  去重 question %d 个中，有**多个不同 answer** 的 = %d（%.1f%%）"
          % (len(q2a), multi, 100.0 * multi / max(len(q2a), 1)))
    print("  ★§64.4 记过：同问题多答案是**刻意设计**（压答案语言模式捷径），不是缺陷")

    # ── 判决 ────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("【判决】")
    print("=" * 78)
    r1 = uq / n
    r2 = usk / n
    print("  S1 question 去重率 = %.4f" % r1)
    print("  S2 骨架去重率     = %.5f（骨架数 %d）" % (r2, usk))
    if usk < 1000:
        print("  ⇒ 骨架只有 %d 个 ⇒ **数据高度饱和**：模型学完这些句式后，"
              % usk)
        print("     剩下 %d 行都是重复 ⇒ 加训练量/改架构都无效。" % (n - usk))
        print("     对策是**扩数据多样性**（下真语料，而非更多同类样本）。")
    elif usk < 10000:
        print("  ⇒ 骨架 %d 个，中等饱和：句式有一定多样性但可能仍不够。" % usk)
    else:
        print("  ⇒ 骨架 %d 个（>=1万）⇒ **数据不饱和**，句式足够多样。" % usk)
        print("     「数据饱和」这个假设在此判据下**不成立**，得另找原因。")

    facts = dict(n_rows=n, uniq_question=uq, uniq_ratio_q=round(r1, 5),
                 uniq_skeleton=usk, uniq_ratio_skel=round(r2, 6),
                 top_q=c.most_common(5), top_skel=top_s,
                 q_with_multi_answer=multi, q_total=len(q2a),
                 corpus=os.path.basename(CORPUS))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    print("  ★只读语料，单次统计，无抽样 ⇒ 这些是**全量精确值**，不是抽样估计。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
