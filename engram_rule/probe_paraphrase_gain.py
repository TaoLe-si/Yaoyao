"""§69.48 静态验证 E1：**同义改写模板**能把 held 骨架命中率从 2.35% 提到多少。

## 为什么先静态验证（不直接重训）

§69.47 实测：骨架命中的 held 样本 F1 = **0.4178**，未命中只有 0.3036（差 37.6%）。
⇒ 可迁移模式在起作用，瓶颈是覆盖面只有 **2.35%**。

对策 E1 = 对 train 做同义问法改写，扩覆盖面。但**改写模板够不够**是未知的：
- 若提到 20%+ ⇒ 值得重训（held F1 会跟着涨）
- 若只到 5% ⇒ 模板不足，得走 E2 下语料

这个数**纯静态就能测**（改写信源 → 重算骨架命中），不用训练、不用 GPU。

## 做法

对每条 train 问句，用一组**同义前缀改写**生成变体，把变体的骨架并进 train 骨架集合，
再看 held 命中率涨多少。

★改写必须是**可复现的字面规则**（不靠模型生成），否则下次复现不出来。
★只改**问句前缀/谓语**，不动答案 —— 否则会改掉 ground truth。

用法：`python probe_paraphrase_gain.py`（纯静态，秒级）
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_scaling_grid_v2 as E2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paraphrase_gain.json")

SLOTS = [
    (re.compile(r"\d+(?:\.\d+)?"), "§"),
    (re.compile(r"《[^》]*》"), "§"),
    (re.compile(r"[A-Za-z][A-Za-z0-9.\-]*"), "§"),
    (re.compile(r"[一-鿿]{2,4}(?:公司|集团|品牌|大学|学院|银行|医院)"), "§"),
]

# ★同义改写表：左为原短语，右为改写列表。只动**前缀/谓语**，不动宾语。
#   全部字面规则，可复现。
PARA = [
    ("你能否", ["你能", "你可以", "能不能", "可以", "能否"]),
    ("你能", ["你能否", "你可以", "能不能", "可以", "能否"]),
    ("你可以", ["你能否", "你能", "能不能", "可以", "能否"]),
    ("能否", ["可以", "能不能", "能够"]),
    ("可以", ["能否", "能不能", "能够"]),
    ("什么是", ["什么叫做", "何为", "啥是"]),
    ("是什么", ["是什么", "为何", "是啥"]),
    ("怎么", ["如何", "怎样", "该如何"]),
    ("如何", ["怎么", "怎样", "该如何"]),
    ("怎样", ["怎么", "如何", "该如何"]),
    ("为什么", ["为何", "怎么会", "什么原因"]),
    ("为何", ["为什么", "怎么会", "什么原因"]),
    ("哪些", ["什么", "有哪", "哪一些"]),
    ("哪家", ["哪些", "什么", "哪个"]),
    ("哪个", ["哪些", "什么", "哪家"]),
    ("请", ["麻烦你", "劳驾", "请你"]),
    ("解释", ["说明", "讲解", "阐述"]),
    ("介绍", ["讲解", "说明", "阐述"]),
    ("列举", ["列出", "举出", "说出"]),
    ("提供", ["给出", "给出一些", "介绍"]),
    ("建议", ["提议", "推荐", "意见"]),
    ("推荐", ["建议", "提议", "介绍"]),
]


def skel(t):
    s = t
    for p, r in SLOTS:
        s = p.sub(r, s)
    return s


def para_variants(q):
    """对一条问句生成同义变体。最多 MAXV 个，避免组合爆炸。"""
    out = [q]
    for src, dsts in PARA:
        if src in q:
            for dst in dsts:
                out.append(q.replace(src, dst, 1))
    return out[:12]           # ★硬上限（AGENTS 规则 13）


def main():
    print("=" * 78)
    print("§69.48 同义改写能把 held 骨架命中率提到多少（纯静态）")
    print("=" * 78)
    E2.load_all()
    rows = E2.load_rows()
    tr, hd = E2.split_rows(rows)
    print("  train %d / held %d" % (len(tr), len(hd)))
    print("  改写规则：%d 组同义前缀，每组最多生成 5 变体，每句上限 12" % len(PARA))

    # ── 基线：不改写的命中率 ────────────────────────────────────────
    base = {skel(r["question"]) for r in tr}
    sk_hd = [skel(r["question"]) for r in hd]
    n_base = sum(1 for s in sk_hd if s in base)
    print()
    print("【基线】不改写")
    print("  train 骨架 %d 个；held 命中 %d / %d = %.2f%%"
          % (len(base), n_base, len(sk_hd), 100.0 * n_base / len(sk_hd)))

    # ── 加改写变体 ──────────────────────────────────────────────────
    aug = set(base)
    n_new = 0
    for r in tr:
        for v in para_variants(r["question"]):
            s = skel(v)
            if s not in aug:
                aug.add(s)
                n_new += 1
    n_aug = sum(1 for s in sk_hd if s in aug)
    print()
    print("【加同义改写】")
    print("  新增骨架 %d 个 ⇒ train 骨架 %d → %d（%.1f 倍）"
          % (n_new, len(base), len(aug), len(aug) / max(len(base), 1)))
    print("  held 命中 %d / %d = %.2f%%（基线 %.2f%%，提升 %.2f 个百分点）"
          % (n_aug, len(sk_hd), 100.0 * n_aug / len(sk_hd),
             100.0 * n_base / len(sk_hd),
             100.0 * (n_aug - n_base) / len(sk_hd)))

    # ── 判决 ────────────────────────────────────────────────────────
    gain = (n_aug - n_base) / len(sk_hd)
    print()
    print("=" * 78)
    print("【判决】")
    print("=" * 78)
    print("  覆盖率：%.2f%% → %.2f%%（+%.2f 个百分点）"
          % (100.0 * n_base / len(sk_hd), 100.0 * n_aug / len(sk_hd), 100 * gain))
    if n_aug / len(sk_hd) > 0.20:
        print("  ⇒ 覆盖率超过 20%% ⇒ **E1 值得做**，重训后期 held F1 应明显上升。")
        print("     按 §69.47 的差分（命中组比未命中高 37.6%%）外推，")
        print("     若覆盖率到 %.0f%%，held F1 有望从 0.30 升到约 %.3f。"
              % (100 * n_aug / len(sk_hd),
                 0.3036 + (0.4178 - 0.3036) * n_aug / len(sk_hd)))
        print("     ★这是**线性外推**，未考虑覆盖率上升会摊薄每个模式的学习强度，")
        print("       只能当量级估计，不当预测。")
    elif n_aug / len(sk_hd) > 0.08:
        print("  ⇒ 覆盖率 %.1f%%，中等 ⇒ 部分有效，但可能仍不够。"
              % (100 * n_aug / len(sk_hd)))
    else:
        print("  ⇒ 覆盖率只到 %.1f%% ⇒ **改写模板不足**，E1 这条路收益有限，" % (100 * n_aug / len(sk_hd)))
        print("     得走 E2（下更多样语料）。")

    facts = dict(train=len(tr), held=len(hd), n_para_rules=len(PARA),
                 base_skel=len(base), aug_skel=len(aug), new_skel=n_new,
                 base_hit=n_base, base_rate=round(n_base / len(sk_hd), 5),
                 aug_hit=n_aug, aug_rate=round(n_aug / len(sk_hd), 5),
                 gain_pp=round(100 * gain, 3),
                 hit_f1=0.4178, unhit_f1=0.3036)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
