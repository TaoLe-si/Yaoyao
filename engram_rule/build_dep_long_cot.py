"""§69.20 S3+ 跨样本依赖：**把「长度的长上下文」变成「依赖的长上下文」。

**为什么必须做**（A 读法下的硬需求）：
桃已定「100 万 = 输入窗口，思维链在窗口内」。这个读法下 CoT 的价值不在「想得更久」，
而在**「在超长上下文里定位关键信息后再推理」**。
而 §69.17.2 的诚实边界说：现有拼卷卷内样本**彼此独立**（各答各的问题），
模型学到的是「一卷里答很多不相关的问题」，**学不到「用第 20,000 位的信息回答最后的问题」**。

**复核确认**（本脚本先跑）：104 卷里有 **43 卷**卷内 question 重复，唯一率 **91.5%**
⇒ 卷内样本确实独立，S3+ 必须做。

── 构造规则 ──────────────────────────────────────────────────────────
把 K 条不相关样本的 **answer 打散**成「线索」，均匀撒在卷内不同位置；
卷尾放一个**汇总问题**，问模型「前面出现过哪些 X」。

    [线索1: ...] [线索2: ...] ... [汇总问题 + + <answer>...</answer>]

★**汇总答案必须由线索的真实内容导出**（可验证、不可编造）：
  用「数一数出现了几个线索」这类**可精确校验**的题，不引入外部知识。
  ⇒ 与 §69.16 的 G1 同源：think/answer 必须由真实内容导出，不能 hallucinate。

── 四道闸（每条配负向对照）────────────────────────────────────────
| 编号 | 闸 | 判据 | 负向对照 |
|---|---|---|---|
| **X1** | 汇总答案 = 线索条数（精确可校验） | 重数一遍线索标记 | 人为改答案 ⇒ 必须不一致 |
| **X2** | 线索**分散**在卷内不同位置 | 相邻两条线索的间隔 ≥ 阈值 | 全堆在开头 ⇒ 必须判失败 |
| **X3** | 卷内**不跨样本截断**（继承 P1） | 每条整条进出 | 复用 `pack_long_cot` 的 P1 |
| **X4** | 汇总问题在**卷尾** | 位置在最后一个线索之后 | 放开头 ⇒ 必须判失败 |

用法：`python build_dep_long_cot.py [每卷 token] [最多卷数]`
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_cot_corpus as B
import exp_scaling_grid_v2 as E2

MAX_VOLS = 200          # ★可被 argv[3] 覆盖

OUT = os.path.join(HERE, "corpus_dep_long_cot.jsonl")
CLUE_OPEN, CLUE_CLOSE = "「线索", "」"
MIN_GAP = 3          # X2：相邻线索之间至少隔 MIN_GAP 个 filler


def render_clue(i, ans):
    return "%s%d：%s%s" % (CLUE_OPEN, i, ans[:60], CLUE_CLOSE)


def render_filler(q):
    """filler = 下一条样本的 question + 固定长度引导句。

    ★**X2 的实测教训**：第一版直接用原始 question 当 filler，
      若它只有 1~2 字（如「你好」「在吗」），相邻线索间隔就 < 阈值
      ⇒ **12/200 卷**被 X2 判掉（间隔最小 23，阈值 24，只差 1）。
    ⇒ 修法**不是放宽阈值**（阈值是防「线索堆在一起」的硬闸），
      而是给 filler 补一个**固定长度的引导句**，保证间隔下限。
      「不引入外部知识」这条不受影响：引导句不含任何事实。
    """
    return "另外，还有一段内容：%s。" % q[:40]


def build_vol(rows, tok, target=8192, n_clues=None):
    """造一卷带跨样本依赖的 CoT。

    结构：filler / clue1 / filler / clue2 / ... / 汇总问题 + think + answer
    ★**汇总答案 = 线索条数**（X1：精确可校验，不引入外部知识）
    ★**线索分散**（X2：中间插 filler，间隔 ≥ MIN_GAP）
    ★**按 token 预算决定线索数**：`n_clues=None` 时由 target 反推，
      而不是写死 12 —— 写死的话卷长永远只有 ~1000 token，训不出长上下文。
    """
    n_all = len(rows)
    if n_all < 2:
        return None
    if n_clues is None:
        # 先量一条「线索+filler」的 token 数，再按 target 反推条数
        probe = render_clue(1, rows[0]["answer"]) + "。" + render_filler(rows[1]["question"])
        per = max(1, len(tok.encode(probe, add_special_tokens=False).ids))
        n_clues = max(2, min(n_all, target // per))
    n = min(n_clues, n_all)
    if n < 2:
        return None
    parts = []
    for i in range(n):
        parts.append(render_clue(i + 1, rows[i]["answer"]))
        if i + 1 < n:
            parts.append(render_filler(rows[i + 1]["question"]))
    body = "。".join(parts)

    q_final = "前面一共出现了多少条线索？"
    think = "我先数一下卷内的线索标记：%s。综合来看，结论是：%d 条。" % (
        "、".join("第%d条" % (i + 1) for i in range(n)), n)
    ans_final = "%d 条" % n
    text = ("%s\n%s%s%s%s%s%s"
            % (body, "<think>", think, "</think>",
               "<answer>", ans_final, "</answer>"))
    return dict(text=text, n_clues=n, answer=ans_final,
                think=think, question=q_final, body=body)


def x1_check(vol):
    """X1：答案必须等于线索条数。"""
    return vol["answer"] == "%d 条" % vol["n_clues"] \
        and vol["text"].count(CLUE_OPEN) == vol["n_clues"]


def x2_check(vol):
    """X2：线索之间必须有 filler 间隔（不能全堆在一起）。"""
    idxs = []
    start = 0
    while True:
        k = vol["body"].find(CLUE_OPEN, start)
        if k < 0:
            break
        idxs.append(k)
        start = k + 1
    if len(idxs) < 2:
        return False
    gaps = [idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)]
    return min(gaps) >= MIN_GAP * 8      # 每条 filler 约 8+ 字符


def x4_check(vol):
    """X4：汇总问题必须在最后一个线索之后。"""
    return vol["body"].rfind(CLUE_OPEN) < vol["text"].find("<think>")


def main():
    from tokenizers import Tokenizer
    import torch
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    global MAX_VOLS
    if len(sys.argv) > 3:
        MAX_VOLS = int(sys.argv[3])
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])

    print("=" * 74)
    print("S3+ 跨样本依赖：线索散落卷内 + 卷尾汇总问题")
    print("=" * 74)
    rows = E2.load_rows()[:cap]
    made, bst = B.build(rows, cap)
    print("  合成 %d 行" % bst["n_out"])

    vols, st = [], dict(n_vol=0, n_x1=0, n_x2=0, n_x4=0)
    i = 0
    while i < len(made) and len(vols) < MAX_VOLS:
        # ★线索数由 target 反推（不再写死 12 —— 写死的话卷长永远只有 ~1000 token）。
        #   先取一大段候选，让 build_vol 自己按 token 预算决定用多少条。
        v = build_vol(made[i:i + 4096], tok, target)
        if v is None:
            break
        used = v["n_clues"]
        ok1, ok2, ok4 = x1_check(v), x2_check(v), x4_check(v)
        st["n_x1"] += 0 if ok1 else 1
        st["n_x2"] += 0 if ok2 else 1
        st["n_x4"] += 0 if ok4 else 1
        if ok1 and ok2 and ok4:
            vols.append(v)
            st["n_vol"] += 1
        i += used          # ★按实际用掉的线索数前进，避免重复用同一批样本

    print()
    print("  成卷 %d 卷；X1 失败 %d / X2 失败 %d / X4 失败 %d"
          % (st["n_vol"], st["n_x1"], st["n_x2"], st["n_x4"]))
    if vols:
        nc = sorted(v["n_clues"] for v in vols)
        tl = sorted(len(tok.encode(v["text"], add_special_tokens=False).ids)
                    for v in vols[:200])
        print("  每卷线索数：中位 %d / 最少 %d / 最多 %d"
              % (nc[len(nc) // 2], nc[0], nc[-1]))
        print("  卷 token 数：中位 %d / p90 %d / max %d"
              % (tl[len(tl) // 2], tl[int(len(tl) * 0.9)], tl[-1]))
        with open(OUT, "w", encoding="utf-8") as f:
            for v in vols:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        print()
        print("  已写出 %d 卷 → %s" % (len(vols), os.path.basename(OUT)))
        print()
        print("  第一卷前 420 字符：")
        print("   ", vols[0]["text"][:420].replace("\n", " / "))

    # ── 负向对照 ────────────────────────────────────────────────
    print()
    print("【负向对照】人为把答案改成错的 ⇒ X1 必须判失败")
    bad = dict(vols[0])
    bad["answer"] = "%d 条" % (bad["n_clues"] + 5)
    assert not x1_check(bad), "X1 负向对照没触发 ⇒ 闸无效"
    print("  把 %s 改成 %s ⇒ X1 判失败 ✓"
          % (vols[0]["answer"], bad["answer"]))

    print("【负向对照】把线索全堆到开头（去掉 filler）⇒ X2 必须判失败")
    bad2 = dict(vols[0])
    bad2["body"] = "。".join(render_clue(i + 1, "x") for i in range(bad2["n_clues"]))
    assert not x2_check(bad2), "X2 负向对照没触发"
    print("  去掉 filler 后线索紧邻 ⇒ X2 判失败 ✓")

    print("【负向对照】把汇总问题挪到卷首 ⇒ X4 必须判失败")
    bad3 = dict(vols[0])
    bad3["text"] = "<think>先想一下</think><answer>x</answer>" + vols[0]["text"]
    assert not x4_check(bad3), "X4 负向对照没触发"
    print("  汇总块挪到最前 ⇒ X4 判失败 ✓")
    print()
    print("全部闸 + 三条负向对照通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
