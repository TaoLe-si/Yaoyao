# -*- coding: utf-8 -*-
"""跨句指代评测（手写 gold，不调用验证器生成）。

两组：
  valid    —— 代词绑定到最近匹配先行词，第二句句义必须精确匹配
  negative —— 无匹配先行词（绑定失败）或绑定成功但语义不成立（验证器拒绝）

另跑一个"绑定第一个匹配"的对照，用来证明"最近"规则真的在起作用。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from discourse import (collect_mentions, process_dialogue,  # noqa: E402
                       process_pair, resolve_candidates)
from lexicon import LEX  # noqa: E402
from semantics import canon, describe  # noqa: E402
from semantics import event, state  # noqa: E402

OUT = os.path.join(HERE, "discourse_results.json")


def E(pred, roles=None, **mods):
    return event(pred, roles or {}, **mods)


def S(pred, theme, **mods):
    return state(pred, {"THEME": theme}, **mods)


VALID = [
    ("小明吃苹果。", "他喝水。", E("喝", {"AGENT": "小明", "PATIENT": "水"})),
    ("老师看见小明。", "他喝水。", E("喝", {"AGENT": "小明", "PATIENT": "水"})),
    ("小明看见老师。", "他喝水。", E("喝", {"AGENT": "老师", "PATIENT": "水"})),
    ("妈妈看见老师。", "他喝水。", E("喝", {"AGENT": "老师", "PATIENT": "水"})),
    ("老师看见妈妈。", "他喝水。", E("喝", {"AGENT": "妈妈", "PATIENT": "水"})),
    ("我买苹果。", "它很红。", S("红", "苹果", DEGREE=True)),
    ("妈妈买香蕉。", "它很甜。", S("甜", "香蕉", DEGREE=True)),
    ("老师看见学校。", "它很大。", S("大", "学校", DEGREE=True)),
    ("妈妈买书。", "它很新。", S("新", "书", DEGREE=True)),
    ("我买苹果和香蕉。", "它很甜。", S("甜", "香蕉", DEGREE=True)),
    ("小明喝水。", "他很高兴。", S("高兴", "小明", DEGREE=True)),
    ("猫吃鱼。", "它喝水。", E("喝", {"AGENT": "鱼", "PATIENT": "水"})),
    ("猫和鱼喝水。", "它很聪明。", S("聪明", "鱼", DEGREE=True)),
    ("妈妈买苹果。", "它很大。", S("大", "苹果", DEGREE=True)),
    ("我买书。", "它很新。", S("新", "书", DEGREE=True)),
]

# 无匹配先行词：绑定必须失败
NO_ANTECEDENT = [
    ("苹果很红。", "他喝水。"),
    ("小明很高兴。", "它很红。"),
    ("今天很热。", "他喝水。"),
    ("桌子在教室。", "他喝水。"),
    (None, "他喝水。"),                 # 单句，无上文
]

# 类型匹配但选择限制不兼容：绑定必须失败（不能用「水」当「聪明」的主题）
INCOMPATIBLE = [
    ("苹果很红。", "它很聪明。"),
    ("小明吃饭。", "它很高兴。"),
]


def _same(a, b):
    return ((a is None and b is None) or
            (a is not None and b is not None and canon(a) == canon(b)))


def main():
    rows = []
    ok = 0
    for s1, s2, gold in VALID:
        pair = process_pair(s1, s2)
        m2 = pair[-1] if len(pair) > 1 else None
        good = _same(m2, gold)
        ok += int(good)
        # 最近 vs 第一个匹配对照
        first = process_pair(s1, s2, nearest=False)[-1]
        first_good = _same(first, gold)
        rows.append({"s1": s1, "s2": s2, "gold": describe(gold),
                     "pred": describe(m2), "ok": good,
                     "first_match_ok": first_good})
        if not good:
            print("MISS", s1, s2, "gold=", describe(gold), "pred=", describe(m2))

    na_rows = []
    for s1, s2 in NO_ANTECEDENT:
        if s1 is None:
            pair = process_dialogue([s2])
            ctx = []
        else:
            pair = process_pair(s1, s2)
            ctx, _ = collect_mentions(s1, [])
        m2 = pair[-1] if len(pair) > 1 else (pair[0] if pair else None)
        resolved = resolve_candidates(s2, ctx)
        na_rows.append({"s1": s1, "s2": s2, "rejected": m2 is None,
                        "binding_failed": len(resolved) == 0})

    sem_rows = []
    for s1, s2 in INCOMPATIBLE:
        pair = process_pair(s1, s2)
        ctx, _ = collect_mentions(s1, [])
        resolved = resolve_candidates(s2, ctx)
        m2 = pair[-1] if len(pair) > 1 else None
        sem_rows.append({"s1": s1, "s2": s2, "rejected": m2 is None,
                         "binding_failed": len(resolved) == 0})

    n_valid = len(VALID)
    nearest_acc = ok / n_valid
    first_acc = sum(1 for r in rows if r["first_match_ok"]) / n_valid
    na_rej = sum(1 for r in na_rows if r["rejected"] and r["binding_failed"]) / len(na_rows)
    sem_rej = sum(1 for r in sem_rows if r["rejected"] and r["binding_failed"]) / len(sem_rows)
    res = {
        "valid_n": n_valid,
        "nearest_accuracy": nearest_acc,
        "first_match_accuracy": first_acc,
        "no_antecedent_reject": na_rej,
        "incompatible_reject": sem_rej,
        "valid_rows": rows,
        "no_antecedent_rows": na_rows,
        "incompatible_rows": sem_rows,
        "note": "gold 手写；指代=最近匹配且满足谓词选择限制；"
                "验证器不是这里的标签来源。",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("valid: nearest=%.4f first-match=%.4f (n=%d)"
          % (nearest_acc, first_acc, n_valid))
    print("no-antecedent reject (binding failed): %.4f (n=%d)"
          % (na_rej, len(na_rows)))
    print("incompatible-antecedent reject (binding failed): %.4f (n=%d)"
          % (sem_rej, len(sem_rows)))
    print("saved %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
