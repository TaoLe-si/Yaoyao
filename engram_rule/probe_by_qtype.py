"""§69.38 按问题类型分桶，定位「答非所问」具体崩在哪一类。

── 为什么先做这个 ──────────────────────────────────────────────────
§69.38 的 M1/M2/M3 已用排除法确认：不是重复、不是说不够长、不是不会收尾。
剩下的是「稳定地生成通顺但答非所问的短文本」。但那是个笼统判断，
**没有指到具体是哪类问题**——而不同问题类型对应完全不同的语料缺口，
改错方向就白改一轮（每轮重训约 2 小时）。

── 怎么做（只读，不生成不训练）────────────────────────────────────
对已落盘的 200 条，按**问题首词的句式特征**打标，然后分桶报 F1 / EM / 生成长度。
分桶规则必须是**可复现的字面规则**，不能靠我人眼判断 —— 否则下次复现不出来。

── 判据 ────────────────────────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **Q1** | 各桶样本数之和 == 200 | 有行没被归类 ⇒ 分桶规则漏了情况 |
| **Q2** | 至少有一桶 F1 显著高于其余 | 若各桶都在 0.2~0.3 ⇒ 不是「某类崩」，是**全面崩**，那归因又要改 |
| **Q3** | 每桶报 n，n<10 的桶标注「样本不足」 | 小样本桶的均值不可当结论（规则 16 同源） |

用法：`python probe_by_qtype.py`（只读已落盘 JSON）
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))

# ★分桶规则：按**字面**关键词匹配，可复现。顺序即优先级（先匹配先归桶）。
BUCKETS = [
    ("是什么/定义", ["是什么", "什么是", "指的是", "定义", "含义", "意思"]),
    ("怎么/如何做", ["怎么", "如何", "怎样"]),
    ("为什么", ["为什么", "为何", "原因"]),
    ("哪些/列举", ["哪些", "有哪", "几种", "多少种", "包括什么", "包括哪些"]),
    ("哪家/哪个", ["哪家", "哪个", "哪位", "哪几"]),
    ("能否/可以", ["能否", "可以", "能不能", "可不可以"]),
    ("推荐/建议", ["推荐", "建议", "帮我"]),
]


def classify(q):
    """按字面规则归类。匹配不到 ⇒ 返回 '其他'（不猜、不归到最近的桶）。"""
    for name, kws in BUCKETS:
        for kw in kws:
            if kw in q:
                return name
    return "其他"


def f1m(rs):
    return sum(r["char_f1"] for r in rs) / len(rs) if rs else float("nan")


def main():
    print("=" * 78)
    print("按问题类型分桶（只读已落盘抽样 200）")
    print("=" * 78)
    print("  分桶规则（字面关键词，可复现）：")
    for name, kws in BUCKETS:
        print("    %-12s %s" % (name, " / ".join(kws)))

    for tag, fn in (("满步 wd=0", "qa_held_s200.jsonl"),
                    ("满步+500步", "qa_held_s200_e500.jsonl")):
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            print("\n[%s] %s 不存在，跳过" % (tag, fn))
            continue
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]

        groups = {}
        for r in rows:
            groups.setdefault(classify(r["question"]), []).append(r)

        n_all = len(rows)
        print()
        print("【%s】n=%d" % (tag, n_all))
        print("  %-12s %5s %7s %7s %9s %9s %s"
              % ("桶", "n", "占比", "EM", "F1", "生成长度中位", "样本不足"))
        order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        for name, rs in order:
            gt = [r["generated_tokens"] for r in rs]
            flag = "  <-- n<10" if len(rs) < 10 else ""
            print("  %-12s %5d %6.1f%% %7d %9.4f %9d%s"
                  % (name, len(rs), 100.0 * len(rs) / n_all,
                     sum(r["em"] for r in rs), f1m(rs), st.median(gt), flag))

        # ── Q1 覆盖性 ────────────────────────────────────────────
        got = sum(len(rs) for _, rs in order)
        print("  ⇒ Q1 覆盖：各桶 n 之和 = %d，应等于 %d  %s"
              % (got, n_all, "OK" if got == n_all else "*不符"))
        assert got == n_all, "Q1 失败：有 %d 行没被归类" % (n_all - got)

        # ── Q2 判别力 ────────────────────────────────────────────
        big = [(nm, rs) for nm, rs in order if len(rs) >= 10]
        if big:
            f1s = [(nm, f1m(rs)) for nm, rs in big]
            f1s.sort(key=lambda kv: -kv[1])
            lo, hi = f1s[0], f1s[-1]
            print("  ⇒ Q2：最好桶 %s = %.4f，最差桶 %s = %.4f，差 %.4f（%.0f%%）"
                  % (hi[0], hi[1], lo[0], lo[1], hi[1] - lo[1],
                     100 * (hi[1] - lo[1]) / max(lo[1], 1e-9)))
            if hi[1] - lo[1] < 0.05:
                print("    ★各桶 F1 都挤在 %.3f~%.3f ⇒ **不是某类崩，是全面崩**"
                      % (lo[1], hi[1]))
            else:
                print("    ⇒ 有可指别的差异 ⇒ 缺口集中在「%s」这类" % lo[0])

    print()
    print("=" * 78)
    print("【边界】")
    print("  · 分桶靠**字面关键词**，问句换种说法就会被归到「其他」—— 它是下界不是精确分类。")
    print("  · n<10 的桶已标注，其均值**不当结论**（规则 16 同源）。")
    print("  · 单 seed、每桶样本更小 ⇒ 全部无种子极差，只作方向指引。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
