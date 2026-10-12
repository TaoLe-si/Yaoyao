"""§69.49 问法重组能把 held 骨架命中率提到多少（纯静态，不用训练）。

## 为什么先测这个（而不是直接下新语料）

§69.48 的实义词读数：held 未命中的 30,413 条里，**核心实义词 79.6% 在 train 问题里出现过**。
⇒ held 问的**知识点** train 基本都见过，缺的是「问法的组合」。

而 §69.47 已实测：句式骨架命中的 held 样本 F1 = **0.4178**，未命中只 0.3036（差 37.6%）。
⇒ 只要把覆盖率提上去，held F1 会跟着涨。

## 做法

把每条 train 问句按「**前缀 + 实体 + 谓词 + 尾缀**」四段拆开，
再交叉重组出变体，把变体骨架并进 train 骨架集合，看 held 命中率涨多少。

★全部字面规则，可复现；只动**问句**，不动答案（否则改掉 ground truth）。
★每句变体数有硬上限（AGENTS 规则 13）。

## 判据（可失败）

| 覆盖率 | 判决 |
|---|---|
| > 15% | **值得重训**：按 §69.47 差分外推 held F1 可到 0.38~0.40 |
| 8~15% | 部分有效，可能仍不够 |
| < 8% | 重组模板不足，需下新语料（E2） |

用法：`python probe_recombine_gain.py`（纯静态，秒级）
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_scaling_grid_v2 as E2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "recombine_gain.json")

SLOTS = [
    (re.compile(r"\d+(?:\.\d+)?"), "§"),
    (re.compile(r"《[^》]*》"), "§"),
    (re.compile(r"[A-Za-z][A-Za-z0-9.\-]*"), "§"),
    (re.compile(r"[一-鿿]{2,4}(?:公司|集团|品牌|大学|学院|银行|医院)"), "§"),
]

# ★谓词同义组：中段替换。上一版（§69.48）只换前缀，所以只 +0.39pp；
#   这次加**中段谓词**，因为差异在中段组合。
PRED = [
    ("包括", ["包含", "由…组成", "涵盖", "有"]),
    ("包含", ["包括", "涵盖", "有"]),
    ("是什么", ["是什么", "是什么", "是什么"]),      # 保持
    ("有哪些", ["有什么", "包括哪些", "包含什么"]),
    ("有什么", ["有哪些", "包括什么", "包含什么"]),
    ("怎么", ["如何", "怎样", "该如何"]),
    ("如何", ["怎么", "怎样", "该如何"]),
    ("怎样", ["怎么", "如何", "该如何"]),
    ("为什么", ["为何", "怎么会", "什么原因"]),
    ("为何", ["为什么", "怎么会", "什么原因"]),
    ("哪家", ["哪些", "什么", "哪个"]),
    ("哪个", ["哪些", "什么", "哪家"]),
    ("解释", ["说明", "讲解", "阐述"]),
    ("介绍", ["讲解", "说明", "阐述"]),
    ("列举", ["列出", "举出", "说出"]),
    ("提供", ["给出", "介绍", "推荐"]),
    ("建议", ["提议", "推荐", "意见"]),
    ("推荐", ["建议", "提议", "介绍"]),
    ("区别", ["差异", "不同", "差别"]),
    ("特点", ["特性", "特征", "特色"]),
    ("作用", ["功能", "用途", "效果"]),
    ("意义", ["含义", "价值", "重要性"]),
]

# ★前/尾缀同义组
PRE = [
    ("你能否", ["你能", "你可以", "能不能", "可以", "能否", "麻烦"]),
    ("你能", ["你能否", "你可以", "能不能", "可以", "能否"]),
    ("你可以", ["你能否", "你能", "能不能", "可以", "能否"]),
    ("请", ["麻烦你", "劳驾", "请你", "帮忙"]),
    ("请问", ["我想问", "问一下", "请教"]),
]

MAXV = 16                 # ★每句变体硬上限


def skel(t):
    s = t
    for p, r in SLOTS:
        s = p.sub(r, s)
    return s


def variants(q):
    """前缀 × 谓词 的笛卡尔变体（截断到 MAXV）。"""
    outs = [q]
    pre_opts = [(q, None)]
    for src, dsts in PRE:
        if src in q:
            pre_opts = [(q.replace(src, d, 1), d) for d in dsts][:4]
            break
    for base, _ in pre_opts:
        for src, dsts in PRED:
            if src in base:
                for d in dsts:
                    if d == "由…组成":
                        cand = base.replace(src, "由", 1) + "组成"
                    else:
                        cand = base.replace(src, d, 1)
                    if cand not in outs:
                        outs.append(cand)
                break
    return outs[:MAXV]


def main():
    print("=" * 78)
    print("§69.49 问法重组（前缀 × 中段谓词）能把覆盖率提到多少")
    print("=" * 78)
    E2.load_all()
    rows = E2.load_rows()
    tr, hd = E2.split_rows(rows)
    print("  train %d / held %d" % (len(tr), len(hd)))
    print("  前缀组 %d 个、谓词组 %d 个，每句变体上限 %d" % (len(PRE), len(PRED), MAXV))

    base = {skel(r["question"]) for r in tr}
    sk_hd = [skel(r["question"]) for r in hd]
    n_base = sum(1 for s in sk_hd if s in base)
    print()
    print("【基线】")
    print("  train 骨架 %d；held 命中 %d / %d = %.2f%%"
          % (len(base), n_base, len(sk_hd), 100.0 * n_base / len(sk_hd)))

    aug = set(base)
    n_new = 0
    n_var = 0
    for r in tr:
        vs = variants(r["question"])
        n_var += len(vs)
        for v in vs:
            s = skel(v)
            if s not in aug:
                aug.add(s)
                n_new += 1
    n_aug = sum(1 for s in sk_hd if s in aug)
    print()
    print("【加重组】")
    print("  平均每句 %.2f 个变体；新增骨架 %s ⇒ train 骨架 %s → %s（%.1f 倍）"
          % (n_var / len(tr), "{:,}".format(n_new), "{:,}".format(len(base)),
             "{:,}".format(len(aug), ), len(aug) / max(len(base), 1)))
    print("  held 命中 %d / %d = %.2f%%（基线 %.2f%%，+%.2f 个百分点）"
          % (n_aug, len(sk_hd), 100.0 * n_aug / len(sk_hd),
             100.0 * n_base / len(sk_hd), 100.0 * (n_aug - n_base) / len(sk_hd)))

    rate = n_aug / len(sk_hd)
    print()
    print("=" * 78)
    print("【判决】")
    print("=" * 78)
    print("  覆盖率 %.2f%% → %.2f%%" % (100.0 * n_base / len(sk_hd), 100 * rate))
    if rate > 0.15:
        print("  ⇒ **>15% ⇒ 值得重训。**")
        print("     按 §69.47 差分（命中 0.4178 vs 未命中 0.3036）线性外推，")
        print("     held F1 有望从 0.3036 升到约 %.3f。"
              % (0.3036 + (0.4178 - 0.3036) * rate))
        print("     ★线性外推**未考虑**覆盖率上升会摊薄每个模式的学习强度，")
        print("       只作量级估计，不当预测。")
    elif rate > 0.08:
        print("  ⇒ 8~15%% ⇒ 部分有效，可能仍不够。")
    else:
        print("  ⇒ <8%% ⇒ 重组模板仍不足，需下新语料（E2）。")

    facts = dict(train=len(tr), held=len(hd),
                 n_pre=len(PRE), n_pred=len(PRED), maxv=MAXV,
                 base_skel=len(base), aug_skel=len(aug), new_skel=n_new,
                 base_hit=n_base, base_rate=round(n_base / len(sk_hd), 5),
                 aug_hit=n_aug, aug_rate=round(rate, 5),
                 gain_pp=round(100 * (n_aug - n_base) / len(sk_hd), 3),
                 hit_f1=0.4178, unhit_f1=0.3036)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
