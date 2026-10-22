# -*- coding: utf-8 -*-
"""开放词表降级验证。

strict=True：词表外直接 raise（训练/检查口径）。
strict=False：连续未知字符合并成一个未知词，作为未知 NP 参与组合，不抛错。

★边界：未知词没有词性/属性，只能占 NP 槽；未知动词无法被识别（没有 POS）。
这是"降级"而不是"理解"——它保证系统不崩，并让选择限制对未知槽位放行。
"""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from learner import predict  # noqa: E402
from lexicon import LEX  # noqa: E402
from parser import parse, tokenize  # noqa: E402
from semantics import describe  # noqa: E402

OUT = os.path.join(HERE, "open_vocab_results.json")
WEIGHTS = os.path.join(HERE, "weights.pkl")

CASES = [
    "我吃苹果",          # 全已知（基线）
    "我吃菠萝",          # 菠萝 OOV，作 PATIENT
    "妈妈买榴莲",        # 榴莲 OOV
    "我喝可乐",          # 可乐 OOV
    "狗吃草莓",          # 草莓 OOV
    "我吃菠萝和苹果",    # OOV 参与并列
    "我游泳",            # OOV 作谓语：没有 POS ⇒ 预期无句法解析
]


def main():
    with open(WEIGHTS, "rb") as f:
        w = pickle.load(f)
    rows = []
    for sent in CASES:
        strict_err = None
        try:
            tokenize(sent, strict=True)
        except ValueError as e:
            strict_err = str(e)
        toks = tokenize(sent, strict=False)
        cands = parse(sent, strict=False)
        pred = None
        if cands:
            pred, _ = predict(w, cands, LEX)
        rows.append({
            "sent": sent,
            "strict_raises": strict_err is not None,
            "open_tokens": toks,
            "n_candidates": len(cands),
            "pred": describe(pred),
            "oov_tokens": [t for t in toks if t.startswith("\x00")],
        })
        print("%-14s strict=%s tokens=%s cands=%d pred=%s"
              % (sent, "raise" if strict_err else "ok", toks, len(cands),
                 describe(pred)))

    res = {"cases": rows,
           "note": "未知词只作为无属性的 NP 放行；未知谓词无法识别（无 POS）。"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
