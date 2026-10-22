# -*- coding: utf-8 -*-
"""独立人工 held-out 评测。

对照两件事：
  1. 模型 vs 手写 gold（真正想知道的）
  2. 手写验证器 vs 手写 gold（验证器是否只是"另一个老师"）
并报告两者在人工集上的分歧条数。
"""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import HELDOUT, TRAIN  # noqa: E402
from learner import predict  # noqa: E402
from lexicon import LEX  # noqa: E402
from manual_heldout import (MANUAL, MANUAL_COVERAGE_GAPS,  # noqa: E402
                            MANUAL_INVALID, MANUAL_VALID)
from parser import parse  # noqa: E402
from semantics import canon, describe  # noqa: E402
from verifier import label_sentence  # noqa: E402

OUT = os.path.join(HERE, "manual_results.json")
WEIGHTS = os.path.join(HERE, "weights.pkl")


def _same(a, b):
    return ((a is None and b is None) or
            (a is not None and b is not None and canon(a) == canon(b)))


def main():
    with open(WEIGHTS, "rb") as f:
        w = pickle.load(f)

    # 防泄漏：人工集不得与 TRAIN/HELDOUT 重叠
    fixed = {s for s, _ in TRAIN} | {s for s, _ in HELDOUT}
    overlap = sorted({s for s, _ in MANUAL} & fixed)
    if overlap:
        raise AssertionError("manual held-out 与 TRAIN/HELDOUT 重叠: %s" % overlap)

    model_rows, ver_rows = [], []
    m_ok = v_ok = 0
    valid_n = invalid_n = 0
    model_ok_valid = ver_ok_valid = 0
    model_ok_invalid = ver_ok_invalid = 0
    for sent, gold in MANUAL:
        cands = parse(sent)
        pred, _ = predict(w, cands, LEX)
        m_good = _same(pred, gold)
        m_ok += int(m_good)
        if gold is None:
            invalid_n += 1
            model_ok_invalid += int(m_good)
        else:
            valid_n += 1
            model_ok_valid += int(m_good)

        vlab = label_sentence(sent, LEX)
        v_good = _same(vlab, gold)
        v_ok += int(v_good)
        if gold is None:
            ver_ok_invalid += int(v_good)
        else:
            ver_ok_valid += int(v_good)

        if not m_good:
            model_rows.append({"sent": sent, "gold": describe(gold),
                               "pred": describe(pred)})
        if not v_good:
            ver_rows.append({"sent": sent, "gold": describe(gold),
                             "verifier": describe(vlab)})

    n = len(MANUAL)
    gaps = []
    for sent, why in MANUAL_COVERAGE_GAPS:
        cands = parse(sent)
        pred, _ = predict(w, cands, LEX)
        vlab = label_sentence(sent, LEX)
        gaps.append({"sent": sent, "why": why,
                     "model": describe(pred), "verifier": describe(vlab)})

    res = {
        "n": n, "valid_n": valid_n, "invalid_n": invalid_n,
        "model": {
            "overall": m_ok / n,
            "valid_exact": model_ok_valid / max(1, valid_n),
            "invalid_reject": model_ok_invalid / max(1, invalid_n),
            "errors": model_rows,
        },
        "verifier_on_manual": {
            "overall": v_ok / n,
            "valid_exact": ver_ok_valid / max(1, valid_n),
            "invalid_reject": ver_ok_invalid / max(1, invalid_n),
            "errors": ver_rows,
        },
        "coverage_gaps": gaps,
        "note": "gold 为手写判断，未调用验证器；句子使用 v0.3 新词，"
                "与 TRAIN/HELDOUT 零重叠。coverage_gaps 不计入准确率，"
                "它们是真汉语成立但当前表示不了的情况。",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("manual held-out: n=%d (valid %d / invalid %d)"
          % (n, valid_n, invalid_n))
    print("model   : overall=%.4f valid=%.4f invalid=%.4f"
          % (res["model"]["overall"], res["model"]["valid_exact"],
             res["model"]["invalid_reject"]))
    print("verifier: overall=%.4f valid=%.4f invalid=%.4f"
          % (res["verifier_on_manual"]["overall"],
             res["verifier_on_manual"]["valid_exact"],
             res["verifier_on_manual"]["invalid_reject"]))
    if model_rows:
        print("-- model errors --")
        for r in model_rows[:12]:
            print("  %s | gold=%s | pred=%s" % (r["sent"], r["gold"], r["pred"]))
    if ver_rows:
        print("-- verifier errors --")
        for r in ver_rows[:12]:
            print("  %s | gold=%s | verifier=%s" % (r["sent"], r["gold"], r["verifier"]))
    print("-- coverage gaps (真汉语成立、当前表示不了；不计入准确率) --")
    for g in gaps:
        print("  %s | %s | model=%s | verifier=%s"
              % (g["sent"], g["why"], g["model"], g["verifier"]))
    print("saved %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
