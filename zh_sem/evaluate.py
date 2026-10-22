# -*- coding: utf-8 -*-
"""评测 v0.1：真实基线 + 留一动词 + 可失败对照。

专家评审 A3/A4/A5：
  - 零训练基线：always-accept / always-reject / cell 查表 / verifier；
  - 留一动词（LOVO）：删掉某动词全部训练句后，测该动词的 held-out；
  - 对照：random-props + 重算 gold（应≈不变）、random-props + 旧 gold（应崩）、
          position_only（去掉动词与填充词特征，应接近 always-accept）。
"""
import json
import os
import pickle
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import HELDOUT, TRAIN, TRAIN_VALID  # noqa: E402
from learner import (evaluate, predict, randomize_props, ratio,  # noqa: E402
                     token_forms, train)
from lexicon import LEX, SENSE_FORM, VERB_FRAME, entity_info  # noqa: E402
from parser import parse  # noqa: E402
from semantics import canon  # noqa: E402
from verifier import label_sentence, verify  # noqa: E402

WEIGHTS = os.path.join(HERE, "weights.pkl")
OUT = os.path.join(HERE, "eval_results.json")
PRED_OUT = os.path.join(HERE, "eval_predictions.json")


def _fmt(stats):
    v = ratio(stats, "valid")
    i = ratio(stats, "invalid")
    n = stats["valid_n"] + stats["invalid_n"]
    a = (stats["valid_ok"] + stats["invalid_ok"]) / max(1, n)
    return {"valid_exact": v, "invalid_reject": i, "overall": a,
            "valid_ok": stats["valid_ok"], "valid_n": stats["valid_n"],
            "invalid_ok": stats["invalid_ok"], "invalid_n": stats["invalid_n"]}


# ---------- 零训练基线 ----------

def baseline_accept_all(data):
    return {"valid_ok": sum(1 for _, g in data if g is not None),
            "valid_n": sum(1 for _, g in data if g is not None),
            "invalid_ok": 0,
            "invalid_n": sum(1 for _, g in data if g is None), "items": []}


def baseline_reject_all(data):
    return {"valid_ok": 0,
            "valid_n": sum(1 for _, g in data if g is not None),
            "invalid_ok": sum(1 for _, g in data if g is None),
            "invalid_n": sum(1 for _, g in data if g is None), "items": []}


def _cell_keys(m, lex):
    keys = []
    for role, head in m["roles"].items():
        info = entity_info(lex, head)
        keys.append((m["pred"], role, info.get("cls") if info else None))
    return keys


def cell_table(data, lex):
    pos, neg = defaultdict(int), defaultdict(int)
    for sent, gold in data:
        cands = parse(sent)
        if gold is None:
            for m in cands:
                for k in _cell_keys(m, lex):
                    neg[k] += 1
        else:
            for m in cands:
                if canon(m) == canon(gold):
                    for k in _cell_keys(m, lex):
                        pos[k] += 1
    return pos, neg


def baseline_cell(data, lex):
    pos, neg = cell_table(TRAIN, lex)
    stats = {"valid_ok": 0, "valid_n": 0, "invalid_ok": 0, "invalid_n": 0, "items": []}
    for sent, gold in data:
        best = None
        for m in parse(sent):
            votes = [pos[k] - neg[k] for k in _cell_keys(m, lex)]
            if all(v >= 0 for v in votes):      # 未见 cell 默认放行（v=0）
                best = m
                break
        if gold is None:
            stats["invalid_n"] += 1
            stats["invalid_ok"] += int(best is None)
        else:
            stats["valid_n"] += 1
            stats["valid_ok"] += int(best is not None and canon(best) == canon(gold))
    return stats


def baseline_verifier(data, lex):
    stats = {"valid_ok": 0, "valid_n": 0, "invalid_ok": 0, "invalid_n": 0, "items": []}
    for sent, gold in data:
        labeled = label_sentence(sent, lex)
        if gold is None:
            stats["invalid_n"] += 1
            stats["invalid_ok"] += int(labeled is None)
        else:
            stats["valid_n"] += 1
            stats["valid_ok"] += int(labeled is not None and canon(labeled) == canon(gold))
    return stats


# ---------- 留一动词 ----------

def leave_one_verb_out(lex):
    rows = []
    for verb_form in sorted({SENSE_FORM[s] for s in VERB_FRAME}):
        sub = [(s, g) for s, g in TRAIN if verb_form not in token_forms(s)]
        if len(sub) == len(TRAIN):
            continue
        hold = [(s, g) for s, g in HELDOUT if verb_form in token_forms(s)]
        if not hold:
            rows.append({"verb": verb_form, "heldout": 0, "note": "held-out 无该动词"})
            continue
        w = train(sub, lex, epochs=40, seed=0)
        st = evaluate(w, hold, lex)
        rows.append({"verb": verb_form, "heldout": len(hold),
                     "valid_exact": ratio(st, "valid"),
                     "invalid_reject": ratio(st, "invalid"),
                     "overall": (st["valid_ok"] + st["invalid_ok"]) / len(hold),
                     "items": st["items"]})
    return rows


def main():
    with open(WEIGHTS, "rb") as f:
        w = pickle.load(f)

    full = evaluate(w, HELDOUT, LEX, verbose=True)
    f = _fmt(full)
    print("== full model (v0.1 逐槽位否决) ==")
    print("  valid exact : %d/%d = %.4f" % (full["valid_ok"], full["valid_n"], f["valid_exact"]))
    print("  invalid rej : %d/%d = %.4f" % (full["invalid_ok"], full["invalid_n"], f["invalid_reject"]))
    print("  overall     : %.4f" % f["overall"])
    for sent, g, p in full["errors"]:
        print("   ERR %s | gold=%s | pred=%s" % (sent, g, p))

    acc = _fmt(baseline_accept_all(HELDOUT))
    rej = _fmt(baseline_reject_all(HELDOUT))
    cell = _fmt(baseline_cell(HELDOUT, LEX))
    ver = _fmt(baseline_verifier(HELDOUT, LEX))
    print("== zero-training baselines ==")
    for name, s in (("accept-all", acc), ("reject-all", rej),
                    ("cell-lookup", cell), ("verifier", ver)):
        print("  %-12s valid=%.4f invalid=%.4f overall=%.4f"
              % (name, s["valid_exact"], s["invalid_reject"], s["overall"]))

    # 对照 1：随机属性 + 重算 gold（自洽世界，应≈不变）
    lex_r, changed = randomize_props(LEX, seed=7)
    train_r = [(s, label_sentence(s, lex_r)) for s, _ in TRAIN]
    hold_r = [(s, label_sentence(s, lex_r)) for s, _ in HELDOUT]
    w_rc = train(train_r, lex_r, epochs=40, seed=0)
    st_rc = evaluate(w_rc, hold_r, lex_r)
    rc = _fmt(st_rc)
    print("== control: random props + regenerated gold (changed %d nouns) ==" % changed)
    print("  valid=%.4f invalid=%.4f overall=%.4f (train gold 正例 %d/%d)"
          % (rc["valid_exact"], rc["invalid_reject"], rc["overall"],
             sum(1 for _, g in train_r if g is not None), len(train_r)))

    # 对照 2：随机属性 + 旧 gold（故意制造标签矛盾，应崩）
    w_rb = train(TRAIN, lex_r, epochs=40, seed=0)
    rb = _fmt(evaluate(w_rb, HELDOUT, lex_r))
    print("== control: random props + old gold (label-inconsistent) ==")
    print("  valid=%.4f invalid=%.4f overall=%.4f"
          % (rb["valid_exact"], rb["invalid_reject"], rb["overall"]))

    # 对照 3：纯位置特征
    w_p = train(TRAIN, LEX, mode="position_only", epochs=40, seed=0)
    po = _fmt(evaluate(w_p, HELDOUT, LEX, mode="position_only"))
    print("== control: position-only (no verb / no filler) ==")
    print("  valid=%.4f invalid=%.4f overall=%.4f"
          % (po["valid_exact"], po["invalid_reject"], po["overall"]))

    lovo = leave_one_verb_out(LEX)
    print("== leave-one-verb-out ==")
    for r in lovo:
        if r["heldout"] == 0:
            print("  %-3s heldout=0 (%s)" % (r["verb"], r.get("note", "")))
        else:
            print("  %-3s heldout=%2d valid=%.2f invalid=%.2f overall=%.2f"
                  % (r["verb"], r["heldout"], r["valid_exact"],
                     r["invalid_reject"], r["overall"]))

    res = {
        "full": f,
        "baselines": {"accept_all": acc, "reject_all": rej,
                      "cell_lookup": cell, "verifier": ver},
        "controls": {
            "random_props_consistent": dict(rc, changed_nouns=changed),
            "random_props_broken": rb,
            "position_only": po,
        },
        "lovo": lovo,
        "predictions": {
            "P1_valid_exact_ge_0.80": f["valid_exact"] >= 0.80,
            "P2_invalid_reject_ge_0.80": f["invalid_reject"] >= 0.80,
            "P2b_beats_accept_all_on_invalid": f["invalid_reject"] > acc["invalid_reject"],
            "P2c_beats_cell_on_invalid": f["invalid_reject"] >= cell["invalid_reject"],
            # ★P3 原判据作废：v0 的 random_class 只换类名、不重算 gold，
            #   掉分测的是标签矛盾。v0.1 拆成两个方向性判据：
            "P3a_consistent_world_invalid_stable":
                abs(rc["invalid_reject"] - f["invalid_reject"]) <= 0.10,
            "P3b_broken_labels_invalid_drop_ge_0.20":
                (f["invalid_reject"] - rb["invalid_reject"]) >= 0.20,
            "P3c_consistent_world_overall_delta": rc["overall"] - f["overall"],
            "P4_position_only_near_accept_all": abs(po["overall"] - acc["overall"]) <= 0.10,
            "P5_lovo_unseen_verb_ge_0.60": all(
                r.get("overall", 0) >= 0.60 for r in lovo if r["heldout"] >= 4),
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    with open(PRED_OUT, "w", encoding="utf-8") as fh:
        json.dump(full["items"], fh, ensure_ascii=False, indent=2)
    print("\nP1 %(P1_valid_exact_ge_0.80)s | P2 %(P2_invalid_reject_ge_0.80)s | "
          "P2b %(P2b_beats_accept_all_on_invalid)s | P2c %(P2c_beats_cell_on_invalid)s"
          % res["predictions"])
    print("P3a %(P3a_consistent_world_invalid_stable)s | "
          "P3b %(P3b_broken_labels_invalid_drop_ge_0.20)s | "
          "P3c_delta=%(P3c_consistent_world_overall_delta)+.4f" % res["predictions"])
    print("P4 %(P4_position_only_near_accept_all)s | P5 %(P5_lovo_unseen_verb_ge_0.60)s"
          % res["predictions"])
    print("saved %s / %s" % (OUT, PRED_OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
