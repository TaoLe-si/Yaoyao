"""§69.46 分组验证「槽位值碎片化」假设：有槽位值的样本是不是明显更差。

── 桃的假设（2026-10-12）─────────────────────────────────────────────
「不是，我说的是没有足够的空间给模型学了。」

实测支撑（`probe_data_saturation.py`）：
  · 骨架去重 143,315 / question 去重 0.9313 ⇒ **句式不饱和**（这个假设被否）
  · 但**槽位值**去重：书名号内容 4,715、英文词 7,138、数字 806
  · 而 **vocab 只有 1024**

⇒ 模型见过 4,715 种书名号内容、7,138 种英文词，但词表只有 1024 个 token
  ⇒ 这些值在 BPE 里被切成**子词碎片**，模型无法把「《三体》」当成一个单位输出。

── 判据（可证伪，先登记）───────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **V1** | 「参考答案不含槽位值」组的 char-F1 **显著高于** 「含槽位值」组 | 若接近 ⇒ 槽位碎片化不是原因，回到 §69.44 跨注意力 |
| **V2** | 两组的 n 都报 | 小样本不当结论（规则 16 同源） |
| **V3** | 两组用**同一批 200 行**（同源自检） | 不同源 ⇒ 不做配对 |

★「含槽位值」的判据：参考答案里出现 `《…》` 或**长度>=3 的英文词**或**数字**。
  纯中文常见词构成的答案归入「不含」。

用法：`python probe_slot_fragment.py`（只读已落盘的 200 条问答）
"""
import io
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
QA = os.path.join(HERE, "qa_held_s200.jsonl")
OUT = os.path.join(HERE, "slot_fragment.json")

PAT_BOOK = re.compile(r"《[^》]*》")
PAT_ENG = re.compile(r"[A-Za-z][A-Za-z0-9.\-]{2,}")
PAT_NUM = re.compile(r"\d")


def has_slot(text):
    return bool(PAT_BOOK.search(text) or PAT_ENG.search(text) or PAT_NUM.search(text))


def f1m(rs):
    return sum(r["char_f1"] for r in rs) / len(rs) if rs else float("nan")


def main():
    print("=" * 78)
    print("槽位值碎片化：有槽位值 vs 无槽位值的生成质量")
    print("=" * 78)
    if not os.path.exists(QA):
        print("缺 %s，先跑 run_qa_tables.py" % os.path.basename(QA))
        return 1

    with io.open(QA, encoding="utf-8") as f:
        qa = [json.loads(l) for l in f if l.strip()]

    # V3 同源自检
    print("  V3 同源：%s（seed=12345 / n=200 / held，由 run_qa_tables 落盘）"
          % os.path.basename(QA))
    print("  行数 = %d" % len(qa))

    with_slot = [r for r in qa if has_slot(r["reference"])]
    no_slot = [r for r in qa if not has_slot(r["reference"])]

    print()
    print("【分组】")
    print("  有槽位值（《…》/ 英文>=3 / 数字） n=%-4d (%4.1f%%)  F1 mean %.4f"
          % (len(with_slot), 100.0 * len(with_slot) / len(qa), f1m(with_slot)))
    print("  无槽位值（纯中文常见词）         n=%-4d (%4.1f%%)  F1 mean %.4f"
          % (len(no_slot), 100.0 * len(no_slot) / len(qa), f1m(no_slot)))

    # V2 小样本标注
    for tag, rs in (("有槽位值", with_slot), ("无槽位值", no_slot)):
        if len(rs) < 10:
            print("  ★V2: %s 组 n=%d < 10 ⇒ 其均值**不当结论**" % (tag, len(rs)))

    d = f1m(no_slot) - f1m(with_slot)
    print()
    print("  V1: F1 差（无 − 有）= %+.4f（无槽位值组%s %.1f%%）"
          % (d, "高" if d > 0 else "低", 100 * abs(d) / max(f1m(with_slot), 1e-9)))

    # 附带：EM 与 LCP=0
    print()
    print("  附带读数：")
    for tag, rs in (("有槽位值", with_slot), ("无槽位值", no_slot)):
        if not rs:
            continue
        print("    %-8s EM=%d/%d  LCP=0 的 %d（%.1f%%）"
              % (tag, sum(r["em"] for r in rs), len(rs),
                 sum(1 for r in rs if r["char_f1"] < 0.35),
                 100.0 * sum(1 for r in rs if r["char_f1"] < 0.35) / len(rs)))

    print()
    print("=" * 78)
    print("【判决】")
    print("=" * 78)
    if d > 0.05:
        print("  ⇒ **无槽位值组明显更好** ⇒ 槽位碎片化是生成差的一个真实成因。")
        print("     vocab=1024 装不下 4,715 种书名号内容 + 7,138 种英文词 ⇒")
        print("     实体被切成子词碎片，模型无法整体输出。")
        print("     ⇒ 对策是**扩词表**（1024 → 8K/16K），而不是先做跨注意力。")
    elif d < -0.05:
        print("  ⇒ 有槽位值组反而更好 ⇒ 假设方向相反，归因需再改。")
    else:
        print("  ⇒ 两组 F1 接近（差 %+.4f）⇒ **槽位碎片化不是主因**，" % d)
        print("     回到 §69.44 的跨注意力方案。")

    facts = dict(qa=os.path.basename(QA), n=len(qa),
                 with_slot=dict(n=len(with_slot), f1=f1m(with_slot),
                                em=sum(r["em"] for r in with_slot)),
                 no_slot=dict(n=len(no_slot), f1=f1m(no_slot),
                              em=sum(r["em"] for r in no_slot)),
                 delta=d,
                 vocab_limit=1024,
                 book_uniq=4715, eng_uniq=7138, num_uniq=806)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    print("  ★单 seed、200 条、分组后样本更小 ⇒ 无种子极差，只作方向指引（规则 16）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
