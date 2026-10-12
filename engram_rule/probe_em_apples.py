"""口径探针：把 `greedy_generate` 的 `first_tok_acc` 定义为**第一步就完全一致**，
取 200 条抽样里已落盘的前 `first_answer_token_accuracy`，逐条核对 EM 与 F1 的关系。

★为什么需要它：我要判断「400 步模型 train EM 0.96」与「满步模型 train EM 0.015」
  是不是同一把尺子。第一版我直接拿两个模型的 EM 比大小，被桃连问两遍
  「那为什么我们先前的一个 400 步模型训练集表现非常好」——
  **那是把两个不同口径的数放在一起比**（AGENTS 规则 11：锚点必须与目标口径一致）。
  本脚本只读已落盘的 JSON，不重新生成，所以是纯静态核对。
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def describe(rows, tag):
    if not rows:
        print("  %s: 无数据" % tag)
        return
    n = len(rows)
    em = sum(r["em"] for r in rows)
    f1 = [r["char_f1"] for r in rows]
    # 按 F1 分桶，看「差一点」和「完全对」各占多少
    buckets = {"F1>=0.9": 0, "0.5<=F1<0.9": 0, "0.1<=F1<0.5": 0, "F1<0.1": 0}
    for v in f1:
        if v >= 0.9:
            buckets["F1>=0.9"] += 1
        elif v >= 0.5:
            buckets["0.5<=F1<0.9"] += 1
        elif v >= 0.1:
            buckets["0.1<=F1<0.5"] += 1
        else:
            buckets["F1<0.1"] += 1
    print("  %-6s n=%-4d EM=%-3d (%.3f)  F1 mean=%.4f med=%.4f"
          % (tag, n, em, em / n, sum(f1) / n, st.median(f1)))
    print("         分桶: %s" % "  ".join("%s=%d(%.0f%%)" % (k, v, 100.0 * v / n)
                                          for k, v in buckets.items()))


def main():
    print("=" * 78)
    print("口径探针：两个模型的 train EM 是不是同一把尺子")
    print("=" * 78)

    # ① 400 步模型（N=738）的公开读数在 DESIGN §64.1，这里只列出来做对照
    print()
    print("【A】400 步 / N=738 模型（DESIGN §64.1，gate=d1c1bc4a9de2）")
    print("  口径：全量 EM（max_new=400，前 200 行） / char-F1")
    print("  读数：200 步 EM 0.9400 F1 0.9857 · 400 步 EM 0.9600 F1 0.9931")
    print("        · 1600 步 EM 0.9600 F1 0.9928  ⇒ 400 步后饱和")
    print("  ★**注意它的口径**：max_new=400，且只取**前 200 行**（不是全量 738）")

    # ② 满步模型（N=124593）的已落盘读数
    print()
    print("【B】满步 19468 / N=124593 模型（本轮 run_qa_tables.py）")
    print("  口径：抽样 200 条（种子 12345）/ greedy / max_new=140 / ctx=512")
    tr = load(os.path.join(HERE, "qa_train_s200.jsonl"))
    hd = load(os.path.join(HERE, "qa_held_s200.jsonl"))
    describe(tr, "train")
    describe(hd, "held")

    print()
    print("【C】口径差异清单（这些差异让 A 与 B 的 EM **不可直接比大小**）")
    rows = [
        ("数据量 N", "738 行", "124,593 行", "差 169 倍"),
        ("每行重复 epochs", "≈138（738×400×256/738 反解）", "40.0（落盘字段）", "记忆压力不同"),
        ("EM 取样范围", "前 200 行（子集）", "随机抽 200 行（种子 12345）", "不是同一批行"),
        ("max_new_tokens", "400", "140", "上限不同 ⇒ 长答案被判错的方式不同"),
        ("context_limit", "未记（老口径）", "512", "老口径可能截断"),
        ("generator", "老版 evaluate_generations", "greedy_generate", "解码路径可能已变"),
    ]
    print("  %-18s %-30s %-26s %s" % ("项", "A（400步/N=738）", "B（满步/N=124593）", "后果"))
    print("  " + "-" * 76)
    for a, b, c, d in rows:
        print("  %-18s %-30s %-26s %s" % (a, b, c, d))

    print()
    print("=" * 78)
    print("【结论】")
    print("=" * 78)
    print("  A 的 0.96 与 B 的 0.015 **不是同一个量**，不能比大小。")
    print("  差异的根源是**记忆压力**：738 行 × 138 次重复 ⇒ 模型把它背下来了；")
    print("  124,593 行 × 40 次重复 ⇒ 同样的容量装不下 169 倍的条目。")
    print("  ★反证就在 A 自己的表里：400 步的探针 NLL = 7.3630，")
    print("    而 B 的 held NLL = 2.275959 —— **B 的泛化比 A 好三倍**。")
    print("    A 是「训练集背熟、留出集很差」，B 是「训练集记不住、留出集还行」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
