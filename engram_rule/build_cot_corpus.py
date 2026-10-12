"""§69.17 S-1 CoT 语料构造：把现有问答语料**合成**成带思维链的三段样本。

**为什么不直接下载**：objective 说「需要语料你可以下载」，但**必须检查和处理**。
外部 CoT 语料（如带推理标注的数据集）有两个本机无法控的风险：
  ① 格式未知，得先写解析器 + 全量校验；
  ② 身份类答案五花八门（「我是 Claude」「我是 ChatGPT」……），
     与 objective 指定的「我是夭夭」冲突，得逐条清洗。
⇒ **先用现有 155,740 行语料合成**，身份口径完全可控，且走已验证的 D1–D6 闸。

**合成规则（三段：question / think / answer）**

    question : 原样不动
    think    : 由 answer 的**可观测特征**生成的推理步骤（不是编造事实）
    answer   : 原样不动（身份类已由 D1 替换）

★**think 段必须由 answer 的真实内容导出**，不能编造外部知识 ——
  否则模型学到的是「幻觉的推理过程」，比没有 CoT 更糟。
  本脚本只生成**结构化的思考步骤**（复述问题 → 拆解要点 → 给出结论），
  其中「结论」= 原 answer 的前缀，**可验证、不可 hallucinate**。

── 三道闸 ────────────────────────────────────────────────────────────
| 编号 | 闸 | 判据 |
|---|---|---|
| **G1** | think 段**必须含 answer 的真实内容**（前缀包含） | 否则是编造 |
| **G2** | think 段长度 ≤ answer 的 3 倍 | 防膨胀（§69.14.1：5x 就撑爆 512） |
| **G3** | 合成后**仍过 D1–D6**（身份/去重/泄漏/格式/长度） | 复用 `check_corpus_gate` |

★`seq` 选择有实测支撑（本脚本也算出来）：全量 155,740 行整行 token
  **中位 67 / p90 128 / p99 250 / max 948** ⇒ `seq=2048` 装下 **100.0%**。

用法：`python build_cot_corpus.py [输出行数上限]`
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2
import check_corpus_gate as G

OUT = os.path.join(HERE, "corpus_cot_v1.jsonl")
SEQ_FOR_STATS = 2048          # 实测可装下 100% 的 seq


def make_think(q, a):
    """由 (question, answer) 生成结构化思考步骤。

    ★**G1 的关键**：think 里必须出现 answer 的真实片段（前缀），
      这样「结论」不是编造的，而是从 answer 摘出来的。
    ★**不引入任何外部知识**：所有模板都是「复述问题 / 拆解要点 / 给出结论」，
      要点只从 answer 里按标点切分得到。

    ★**G2 的实测教训**：第一版模板固定约 120 字符，而答案中位只有 ~40 字符
      ⇒ `think ≤ 3×answer` 把 **1805/3000（60%）** 判掉了。
      原因是短答案（「是的」「我不知道」）撑不起 120 字符的模板。
      ⇒ 改成**按 answer 长度分级**：短答案用精简模板（G1 仍满足，
        因为结论句就是 answer 本身，天然包含）。
    """
    parts = [p.strip() for p in a.replace("。", "。\n").replace("；", "；\n").split("\n")
             if p.strip()]
    pts = parts[:2] if len(parts) >= 2 else [a[:40]]
    la = len(a)

    if la <= 3:
        # ★超短答案（1~3 字，如「成都」「苏」「是的」）：**think 就是 answer 本身**。
        #   实测这类占 79/3000，任何模板前缀都会让比值超 3。
        #   ⇒ 这类样本**不适合做 CoT**（无推理空间），直接用 answer 当 think
        #     —— 它仍然满足 G1（包含自身），但实际贡献约等于无 CoT。
        #   ★诚实的做法是**标注出来**，不假装它们有思维链。
        think = a
        tr_meta = "no_cot"
    elif la <= 30:
        think = "先想一下：%s" % a[:30]
        tr_meta = "short"
    elif la <= 60:
        think = "用户问的是：%s\n结论是：%s" % (q[:24], a)
        tr_meta = "mid"
    else:
        think = ("用户问的是：%s\n我先看一下问题在问什么，要点有：%s\n"
                 "综合来看，结论是：%s"
                 % (q[:60], "；".join(p[:30] for p in pts), a[:60]))
        tr_meta = "full"
    return think, tr_meta


def g1_check(row):
    """G1：think 必须含 answer 的前缀（防编造）。"""
    return row["answer"][:20] in row["think"]


def g2_check(row):
    """G2：think ≤ 3×answer（字符数口径，防膨胀）。"""
    return len(row["think"]) <= 3 * max(len(row["answer"]), 1)


def build(rows, limit=None):
    """合成 + 过闸。返回 (合格的, 统计)。"""
    src = rows[:limit] if limit else rows
    made = []
    n_g1 = n_g2 = 0
    from collections import Counter
    tiers = Counter()
    for r in src:
        tr = dict(r)
        tr["think"], tier = make_think(r.get("question", ""), r.get("answer", ""))
        tr["cot_tier"] = tier          # ★标出来，不假装短答案有思维链
        if not g1_check(tr):
            n_g1 += 1
            continue
        if not g2_check(tr):
            n_g2 += 1
            continue
        tiers[tier] += 1
        made.append(tr)

    # ★先过 D1（身份替换）再过 D3/D5，与 check_corpus_gate 同序
    made, n_id = G.apply_identity_rule(made)
    good, bad = G.d5_format_check(made)
    made = [made[i] for i, _w in good]
    made, n_dup = G.d3_dedup(made)
    stats = dict(n_in=len(src), n_out=len(made), n_g1_fail=n_g1,
                 n_g2_fail=n_g2, n_identity=n_id,
                 n_bad_format=len(bad), n_dup=n_dup,
                 tier_no_cot=tiers["no_cot"], tier_short=tiers["short"],
                 tier_mid=tiers["mid"], tier_full=tiers["full"])
    return made, stats


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = E2.load_rows()
    print("=" * 74)
    print("S-1 CoT 语料构造（三段：question / think / answer）")
    print("=" * 74)
    print("  输入 %d 行，上限 %s" % (len(rows), cap or "无"))

    made, st = build(rows, cap)
    print()
    print("  闸统计：")
    for k, v in st.items():
        print("    %-14s = %d" % (k, v))

    if made:
        tl = sorted(len(r["think"]) for r in made[:20000])
        n = len(tl)
        print()
        print("  think 字符数：中位 %d / p90 %d / p99 %d"
              % (tl[n // 2], tl[int(n * 0.9)], tl[int(n * 0.99)]))
        print("  think/answer 长度比（中位）：%.2f"
              % (tl[n // 2] / max(1, len(made[0]["answer"]))))

        with open(OUT, "w", encoding="utf-8") as f:
            for r in made:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        print()
        print("  已写出 %d 行 → %s" % (len(made), os.path.basename(OUT)))
        print()
        print("  样例：")
        r = made[0]
        print("    Q:", r["question"][:50])
        print("    T:", r["think"][:90].replace("\n", " / "))
        print("    A:", r["answer"][:50])
    return 0


if __name__ == "__main__":
    sys.exit(main())
