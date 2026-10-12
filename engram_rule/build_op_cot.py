"""§69.22 跨变量运算：把「按名字检索」推到「语义推理」。

**为什么做**：§69.21.3 的诚实边界说 —— 「变量甲 = X」的 X 是原样本 answer，
**不是语义关联**。模型做的是「按名字找到对应的字符串」，**没有跨变量推理**。
那是**检索型**长程依赖（RULER 的 variable-tracking），不是**推理型**。

**这一档的构造**：让答案**依赖两个以上变量的组合**，必须真的「推理」一步：

    「变量甲 = 3 个」 「变量乙 = 5 个」 ... 中间大量 filler ...
    问：「甲和乙一共多少个？」

★**答案 = 甲 + 乙**，是**从两个变量算出来的**，不是任何一个变量的原文。
  ⇒ 模型必须 (1) 检索到甲 (2) 检索到乙 (3) 做加法 (4) 写答案。**四步缺一不可。**
  而 §69.21 只需要 (1)(2) 两步 —— 这就是差别。

★**可精确校验**：用**数字**做变量值，答案就是算术结果，**不引入外部知识**。
  数字从原始样本的 answer 里抽（取第一段连续数字），抽不到就跳过该组。

── 四道闸（每条配负向对照）────────────────────────────────────────
| 编号 | 闸 | 判据 | 负向对照 |
|---|---|---|---|
| **R1** | 答案 = 甲 + 乙（算术正确） | 重新算一遍 | 把答案改成甲+乙+1 ⇒ 必须不一致 |
| **R2** | 甲、乙都必须在卷内 | 都能按名字找到 | 删掉乙 ⇒ 必须判失败 |
| **R3** | 两个赋值都必须在卷前 1/4 | 位置 < len/4 | 挪到卷尾 ⇒ 必须判失败 |
| **R4** | 干扰变量 ≥ 3 | 数量 ≥ 3 | 降到 1 ⇒ 必须判失败 |

用法：`python build_op_cot.py [每卷 token] [最多卷数]`
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_cot_corpus as B
import exp_scaling_grid_v2 as E2

OUT = os.path.join(HERE, "corpus_op_cot.jsonl")
VAR = "「变量"
N_DISTRACTOR = 5
NUM_RE = re.compile(r"\d+")


def first_int(s, lo=2, hi=90):
    """从文本里取第一个落在 [lo,hi] 的整数；取不到返回 None。

    ★**限定范围**：不加限制的话可能取到年份(2024)或版本号(1.16)，
      加出来答案很怪但**仍然可校验** —— 不过为了可读性还是限一下。
    """
    for m in NUM_RE.finditer(s):
        v = int(m.group())
        if lo <= v <= hi:
            return v
    return None


def build_vol(rows, tok, target=8192):
    """造一卷「跨变量运算」的 CoT。"""
    if len(rows) < N_DISTRACTOR + 2:
        return None
    va = first_int(rows[0]["answer"])
    vb = first_int(rows[1]["answer"])
    if va is None or vb is None or va == vb:
        return None

    parts = []
    parts.append("%s甲 = %d 个」" % (VAR, va))
    parts.append("%s乙 = %d 个」" % (VAR, vb))
    names = ["丙", "丁", "戊", "己", "庚"][:N_DISTRACTOR]
    probe = "另外，还有一段内容：%s。" % "内容"
    per = max(1, len(tok.encode(probe, add_special_tokens=False).ids))
    n_fill = max(0, (target - 200) // (N_DISTRACTOR * per))
    for nm, r in zip(names, rows[2:2 + N_DISTRACTOR]):
        for _ in range(n_fill):
            parts.append("另外，还有一段内容：%s。" % r["question"][:40])
        parts.append("%s%s = %d 个」" % (VAR, nm, first_int(r["answer"]) or 7))
    body = "。".join(parts)

    ans = va + vb
    think = ("我先找变量甲和乙的赋值。甲在卷首，是 %d 个；乙紧随其后，是 %d 个。"
             "然后把它们加起来：%d + %d = %d。结论是：%d 个。" % (va, vb, va, vb, ans, ans))
    # ★**又是我写错**：格式串是 `%s...%d...` 共 3 个占位符，参数却是
    #   `(body, think, ans)` 三个里 ans 是 int —— 但 think 是 str，
    #   `%s` 吃 think 没问题，`%d` 该吃 ans…… 报错说 `%d` 拿到 str，
    #   说明我漏了 think 前面的 `%s`（原来只有 2 个占位符）。
    #   改成显式三个占位符：body(→%s) think(→%s) ans(→%d)
    text = ("%s%s<answer>%d 个</answer>" % (body, think, ans))
    return dict(text=text, va=va, vb=vb, answer="%d 个" % ans,
                body=body, n_distractor=N_DISTRACTOR,
                question="变量甲和变量乙一共多少个？")


def r1_check(vol):
    return int(vol["answer"].split()[0]) == vol["va"] + vol["vb"]


def r2_check(vol):
    return ("%s甲 = " % VAR) in vol["body"] and ("%s乙 = " % VAR) in vol["body"]


def r3_check(vol):
    q = len(vol["body"]) // 4
    return vol["body"].find("%s甲 = " % VAR) < q and vol["body"].find("%s乙 = " % VAR) < q


def r4_check(vol):
    return vol["n_distractor"] >= 3


def main():
    from tokenizers import Tokenizer
    import torch
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])

    print("=" * 74)
    print("S3+++ 跨变量运算：答案 = 甲 + 乙，必须检索两个变量再算")
    print("=" * 74)
    rows = E2.load_rows()[:cap]
    made, bst = B.build(rows, cap)
    print("  合成 %d 行" % bst["n_out"])

    vols, st = [], dict(n_vol=0, n_r1=0, n_r2=0, n_r3=0, n_r4=0, n_skip=0)
    step = N_DISTRACTOR + 2
    i = 0
    while i + step <= len(made) and len(vols) < 200:
        grp = made[i:i + step]
        v = build_vol(grp, tok, target)
        if v is None:
            st["n_skip"] += 1
            i += step
            continue
        oks = (r1_check(v), r2_check(v), r3_check(v), r4_check(v))
        for k, ok in zip(("n_r1", "n_r2", "n_r3", "n_r4"), oks):
            st[k] += 0 if ok else 1
        if all(oks):
            vols.append(v)
            st["n_vol"] += 1
        i += step

    print()
    print("  成卷 %d 卷（跳过 %d 组：抽不到两个不同数字）" % (st["n_vol"], st["n_skip"]))
    print("  R1 失败 %d / R2 失败 %d / R3 失败 %d / R4 失败 %d"
          % (st["n_r1"], st["n_r2"], st["n_r3"], st["n_r4"]))
    if vols:
        tl = sorted(len(tok.encode(v["text"], add_special_tokens=False).ids)
                    for v in vols[:200])
        print("  卷 token 数：中位 %d / p90 %d / max %d"
              % (tl[len(tl) // 2], tl[int(len(tl) * 0.9)], tl[-1]))
        with open(OUT, "w", encoding="utf-8") as f:
            for v in vols:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        print("  已写出 %d 卷 → %s" % (len(vols), os.path.basename(OUT)))
        v0 = vols[0]
        print()
        print("  第一卷：甲=%d 乙=%d 答案=%d 个" % (v0["va"], v0["vb"], v0["va"] + v0["vb"]))
        print("    卷首 90 字:", v0["body"][:90])
        print("    卷尾 190 字:", v0["text"][-190:].replace("\n", " "))

    print()
    print("【负向对照】把答案改成 甲+乙+1 ⇒ R1 必须判失败")
    if vols:
        bad = dict(vols[0]); bad["answer"] = "%d 个" % (bad["va"] + bad["vb"] + 1)
        assert not r1_check(bad), "R1 负向对照没触发"
        print("  答案被 +1 ⇒ R1 判失败 ✓")

    print("【负向对照】删掉乙的赋值 ⇒ R2 必须判失败")
    if vols:
        bad2 = dict(vols[0])
        bad2["body"] = bad2["body"].replace("%s乙 = " % VAR, "%sX = ", 1)
        assert not r2_check(bad2), "R2 负向对照没触发"
        print("  乙的赋值被改掉 ⇒ R2 判失败 ✓")

    print("【负向对照】把甲、乙挪到卷尾 ⇒ R3 必须判失败")
    if vols:
        bad3 = dict(vols[0])
        # ★**负向对照写错**：第一版只在 body **前面**塞 40 段 filler。
        #   但 r3_check 找的是 `find()`（**第一个**出现位置），
        #   塞前面确实会让 find 变大 —— 可 body 原本只有 ~200 字，
        #   40 段 filler × 约 30 字 ≈ 1200 字，而阈值是 len(body)//4。
        #   实测 1200 < len//4 不成立 ⇒ 对照没触发。
        #   ⇒ 改成**直接把甲、乙两段移到 body 末尾**，这是真正的「挪到卷尾」。
        import re as _re
        b = bad3["body"]
        m_a = _re.search(r"「变量甲 = \d+ 个」", b)
        m_b = _re.search(r"「变量乙 = \d+ 个」", b)
        if m_a and m_b:
            stripped = b.replace(m_a.group(), "").replace(m_b.group(), "")
            bad3["body"] = stripped + "。" + m_a.group() + "。" + m_b.group()
        assert not r3_check(bad3), "R3 负向对照没触发"
        print("  把甲、乙两段移到 body 末尾 ⇒ R3 判失败 ✓")

    print("【负向对照】干扰变量降到 1 ⇒ R4 必须判失败")
    if vols:
        bad4 = dict(vols[0]); bad4["n_distractor"] = 1
        assert not r4_check(bad4), "R4 负向对照没触发"
        print("  干扰降到 1 ⇒ R4 判失败 ✓")
    print()
    print("  ★这一档与 69.21 的差别：必须**检索两个变量再运算**，不是只找一个")
    print("全部闸 + 四条负向对照通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
