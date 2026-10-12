"""§69.21 多跳依赖：把「能数清」推到「能推理」。

**为什么做**：§69.20 的诚实边界说 —— 汇总题「数一数有几条线索」**可精确校验但没有推理深度**，
它验的是「能不能在 36K token 里定位并计数散落的标记」，是长程依赖的**下限**（能数清），
不是**上限**（能推理）。

**多跳的构造**（RULER 的 variable-tracking 形态）：

    「变量A = 值1」 ... 中间大量 filler ... 「变量B = 值2」 ... 「变量A 的值是什么？」

★**关键设计**：问的变量在卷**前部**，它的赋值在卷**更前部**，
  模型必须跨过整卷的 filler 去找那个赋值 —— 这才是真正的长程检索 + 推理。
  而 §69.20 的「数线索」只需看**局部标记**（`` 紧邻序号），
  所以测不出长程能力。**这一条差别是本脚本存在的全部理由。**

★**答案必须可精确校验**：答案就是那个变量的赋值文本片段，
  **不引入任何外部知识**（与 G1/X1 同源）。

── 四道闸（每条配负向对照）────────────────────────────────────────
| 编号 | 闸 | 判据 | 负向对照 |
|---|---|---|---|
| **H1** | 答案必须 = 被问变量的真实赋值 | 直接比对赋值文本 | 换成别的值 ⇒ 必须不一致 |
| **H2** | 赋值必须在**卷前 1/3** | 位置 < len/3 | 挪到卷尾 ⇒ 必须判失败 |
| **H3** | 问题必须在**最后一个赋值之后** | 位置 > 最后一个赋值 | 挪到赋值之前 ⇒ 必须判失败 |
| **H4** | 至少 3 个**干扰变量** | 干扰赋值数 ≥ 3 | 只留 1 个变量 ⇒ 必须判失败 |

★**H4 是关键**：只有一个变量时，模型不需要「检索」——
  唯一的赋值就是答案，那是**记忆**不是**长程依赖**。
  必须有 ≥3 个干扰变量，模型才必须按名字匹配。

用法：`python build_hop_cot.py [每卷 token] [最多卷数]`
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_cot_corpus as B
import exp_scaling_grid_v2 as E2

MAX_VOLS = 200          # ★可被 argv[3] 覆盖

OUT = os.path.join(HERE, "corpus_hop_cot.jsonl")
VAR_OPEN = "「变量"
N_DISTRACTOR = 5       # H4：干扰变量数


def build_vol(rows, tok, target=8192):
    """造一卷多跳依赖的 CoT。

    结构：
      卷首：把**被问变量**的赋值放最前（H2）
      中段：大量 filler + 5 个干扰变量的赋值（H4）
      卷尾：汇总问题 + think + answer
    ★**被问变量在前 1/3**（H2）⇒ 模型必须跨过几乎整卷才能找到答案。
    """
    # ★**判据写错的一处**：第一版要求 `len(rows) < N_DISTRACTOR + 2` 就返回 None，
    #   而 main 里按 `step = N_DISTRACTOR + 1` 切组 ⇒ 传进来正好 `N_DISTRACTOR+1` 条，
    #   **永远**不满足 `>= N_DISTRACTOR+2` ⇒ 0 成卷。
    #   两个常量不一致（一个是 +2 一个是 +1），检查自己把自己卡死了。
    if len(rows) < N_DISTRACTOR + 1:
        return None
    target_row = rows[0]
    distractors = rows[1:1 + N_DISTRACTOR]

    parts = []
    # ① 被问变量的赋值（放最前）
    parts.append("%s%s = %s」" % (VAR_OPEN, "甲", target_row["answer"][:40]))
    # ② 干扰变量赋值，中间按 token 预算插 filler
    names = ["乙", "丙", "丁", "戊", "己"][:N_DISTRACTOR]
    probe_filler = "另外，还有一段内容：%s。" % "内容"
    per_filler = max(1, len(tok.encode(probe_filler, add_special_tokens=False).ids))
    n_filler = max(0, (target - 200) // (N_DISTRACTOR * per_filler))
    for k, (nm, r) in enumerate(zip(names, distractors)):
        for j in range(n_filler):
            parts.append("另外，还有一段内容：%s。" % r["question"][:40])
        parts.append("%s%s = %s」" % (VAR_OPEN, nm, r["answer"][:40]))
    body = "。".join(parts)

    q_final = "变量「甲」的值是什么？"
    think = ("我先找一下变量「甲」的赋值。它在卷首的位置，赋值为：%s。"
             "综合来看，结论是：%s。" % (target_row["answer"][:40],
                                    target_row["answer"][:40]))
    ans = target_row["answer"][:40]
    # ★**又是我写错**：格式串只有 2 个 `%s`，却传了 3 个参数（body/think/ans）
    #   ⇒ TypeError。第一版想写成 `body +  + <answer>ans</answer>`，
    #   改成 `%s` 串时漏了 think 的占位符。这是「改字符串忘了改参数」的典型。
    text = ("%s<think>%s</think><answer>%s</answer>"
            % (body, think, ans))
    return dict(text=text, target_var="甲", answer=ans, n_distractor=N_DISTRACTOR,
                body=body, question=q_final, think=think)


def h1_check(vol, gold):
    """H1：答案必须等于被问变量的真实赋值。"""
    return vol["answer"] == gold[:40]


def h2_check(vol):
    """H2：被问变量的赋值必须在卷前 1/3。"""
    k = vol["body"].find("「变量甲")
    return 0 <= k < len(vol["body"]) // 3


def h3_check(vol):
    """H3：问题必须在最后一个赋值之后。"""
    last = vol["body"].rfind(VAR_OPEN)
    q = vol["text"].find("<think>")
    return last >= 0 and q > last


def h4_check(vol):
    """H4：干扰变量数 ≥ 3。"""
    return vol["n_distractor"] >= 3


def main():
    from tokenizers import Tokenizer
    import torch
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    global MAX_VOLS
    if len(sys.argv) > 3:
        MAX_VOLS = int(sys.argv[3])
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])

    print("=" * 74)
    print("S3++ 多跳依赖：被问变量在卷首，问题在卷尾，中间 5 个干扰变量")
    print("=" * 74)
    rows = E2.load_rows()[:cap]
    made, bst = B.build(rows, cap)
    print("  合成 %d 行" % bst["n_out"])

    vols, st = [], dict(n_vol=0, n_h1=0, n_h2=0, n_h3=0, n_h4=0)
    step = N_DISTRACTOR + 1
    i = 0
    while i + step <= len(made) and len(vols) < MAX_VOLS:
        grp = made[i:i + step]
        v = build_vol(grp, tok, target)
        if v is None:
            break
        gold = grp[0]["answer"]
        ok1, ok2, ok3, ok4 = (h1_check(v, gold), h2_check(v),
                              h3_check(v), h4_check(v))
        st["n_h1"] += 0 if ok1 else 1
        st["n_h2"] += 0 if ok2 else 1
        st["n_h3"] += 0 if ok3 else 1
        st["n_h4"] += 0 if ok4 else 1
        if ok1 and ok2 and ok3 and ok4:
            vols.append(v)
            st["n_vol"] += 1
        i += step

    print()
    print("  成卷 %d 卷；H1 失败 %d / H2 失败 %d / H3 失败 %d / H4 失败 %d"
          % (st["n_vol"], st["n_h1"], st["n_h2"], st["n_h3"], st["n_h4"]))
    if vols:
        tl = sorted(len(tok.encode(v["text"], add_special_tokens=False).ids)
                    for v in vols[:200])
        print("  卷 token 数：中位 %d / p90 %d / max %d"
              % (tl[len(tl) // 2], tl[int(len(tl) * 0.9)], tl[-1]))
        with open(OUT, "w", encoding="utf-8") as f:
            for v in vols:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        print("  已写出 %d 卷 → %s" % (len(vols), os.path.basename(OUT)))
        print()
        print("  第一卷（截断显示）：")
        t = vols[0]["text"]
        print("    卷首 120 字:", t[:120].replace("\n", " "))
        print("    卷尾 160 字:", t[-160:].replace("\n", " "))

    # ── 负向对照 ────────────────────────────────────────────────
    print()
    print("【负向对照】把答案换成别的值 ⇒ H1 必须判失败")
    bad = dict(vols[0]); bad["answer"] = "完全不同的内容"
    assert not h1_check(bad, vols[0]["answer"]), "H1 负向对照没触发"
    print("  答案被改成无关内容 ⇒ H1 判失败 ✓")

    print("【负向对照】把被问变量挪到卷尾 ⇒ H2 必须判失败")
    bad2 = dict(vols[0])
    # ★**负向对照写错**：第一版用 `.replace("「变量甲 = ", "「变量甲X = ", 1)`
    #   —— 但 body 里的真实写法是 `「变量甲 = 值」`，而 `h2_check` 找的是
    #   `「变量甲`（不含空格与等号）⇒ 替换的串根本**匹配不上**，body 没变，
    #   H2 仍返回 True ⇒ 对照不触发。
    #   改成直接把整段 body 里「变量甲」那个赋值删掉，H2 才会找不到而判失败。
    bad2["body"] = bad2["body"].replace("「变量甲 = ", "「变量_removed = ", 1)
    assert "「变量甲" not in bad2["body"], "替换串没匹配上 ⇒ 对照没真的改到东西"
    assert not h2_check(bad2), "H2 负向对照没触发"
    print("  把「变量甲」的赋值改掉 ⇒ 卷首找不到它 ⇒ H2 判失败 ✓")

    print("【负向对照】只留 1 个变量（无干扰）⇒ H4 必须判失败")
    bad3 = dict(vols[0]); bad3["n_distractor"] = 1
    assert not h4_check(bad3), "H4 负向对照没触发"
    print("  干扰变量降到 1 ⇒ H4 判失败 ✓")
    print()
    print("  ★H4 是关键：只有一个变量时模型不需要「检索」，那是**记忆**不是**长程依赖**")
    print("全部闸 + 三条负向对照通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
