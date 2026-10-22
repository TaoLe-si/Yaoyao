# -*- coding: utf-8 -*-
"""大冻结 held-out（300 条，覆盖 12 类构式）。

★诚实边界：标签由手写验证器生成，不是独立人工真值；这里的读数只能叫
"与验证器一致率"。独立人工标注的 83 条仍是主判据（evaluate.py）。

首次运行生成并落盘 large_heldout.json；之后默认复用同一份（冻结）。
用 --regen 重新生成（会改变评测集，不得与旧读数直接比）。
"""
import json
import os
import pickle
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import HELDOUT, TRAIN  # noqa: E402
from learner import predict  # noqa: E402
from lexicon import LEX, VERB_FRAME  # noqa: E402
from parser import parse  # noqa: E402
from semantics import canon, describe  # noqa: E402
from verifier import label_sentence  # noqa: E402

FROZEN = os.path.join(HERE, "large_heldout.json")
OUT = os.path.join(HERE, "large_eval_results.json")
WEIGHTS = os.path.join(HERE, "weights.pkl")
N_ITEMS = 300


def _classifier(noun):
    cls = LEX[noun].get("cls")
    if cls == "READABLE":
        return "本"
    if cls in ("ANIMAL", "ANIMAL_EDIBLE"):
        return "只"
    return "个"


def generate():
    """按构式均衡采样：每类约 N_ITEMS/12 条，避免组合爆炸与构式失衡。"""
    nouns = [f for f, i in LEX.items() if i["pos"] in ("N", "PRON")]
    adjs = [f for f, i in LEX.items() if i["pos"] == "A"]
    by_arity = {1: [], 2: [], 3: []}
    for f, i in LEX.items():
        if i["pos"] == "V":
            by_arity[len(i["frame"])].append(f)

    def make(tag, rng):
        def v(arity):
            return rng.choice(by_arity[arity])
        n = lambda: rng.choice(nouns)
        a = lambda: rng.choice(adjs)
        if tag == "SV":
            return "%s%s" % (n(), v(1))
        if tag == "SVO":
            return "%s%s%s" % (n(), v(2), n())
        if tag == "SVOO":
            return "%s%s%s%s" % (n(), v(3), n(), n())
        if tag == "BA":
            return "%s把%s%s了" % (n(), n(), v(2))
        if tag == "BEI":
            return "%s被%s%s了" % (n(), n(), v(2))
        if tag == "Q":
            return "%s%s%s吗" % (n(), v(2), n())
        if tag == "CONJ":
            return "%s和%s%s%s" % (n(), n(), v(2), n())
        if tag == "QUANT":
            x = n()
            return "一%s%s%s%s" % (_classifier(x), x, v(2), n())
        if tag == "LOC":
            return "%s在%s" % (n(), n())
        if tag == "ADJ":
            return "%s很%s" % (n(), a())
        if tag == "NEGADJ":
            return "%s不%s" % (n(), a())
        if tag == "ATTR":
            return "%s的%s" % (a(), n())
        raise ValueError(tag)

    tags = ["SV", "SVO", "SVOO", "BA", "BEI", "Q", "CONJ",
            "QUANT", "LOC", "ADJ", "NEGADJ", "ATTR"]
    per = max(1, N_ITEMS // len(tags))
    fixed = {s for s, _ in TRAIN} | {s for s, _ in HELDOUT}
    rng = random.Random(0)
    out = []
    for tag in tags:
        got = 0
        seen = set()
        tries = 0
        while got < per and tries < per * 200:
            tries += 1
            sent = make(tag, rng)
            if sent in fixed or sent in seen:
                continue
            seen.add(sent)
            try:
                if not parse(sent):
                    continue
            except ValueError:
                continue
            gold = label_sentence(sent, LEX)
            out.append({"tag": tag, "sent": sent,
                        "gold": gold, "gold_desc": describe(gold)})
            got += 1
    rng.shuffle(out)
    return out


def load_or_generate(regen=False):
    if os.path.exists(FROZEN) and not regen:
        with open(FROZEN, "r", encoding="utf-8") as f:
            return json.load(f), False
    data = generate()
    with open(FROZEN, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data, True


def main():
    regen = "--regen" in sys.argv
    data, fresh = load_or_generate(regen)
    with open(WEIGHTS, "rb") as f:
        w = pickle.load(f)

    by_tag = {}
    ok = 0
    valid_ok = valid_n = inv_ok = inv_n = 0
    errors = []
    for row in data:
        sent, gold = row["sent"], row["gold"]
        pred, _ = predict(w, parse(sent), LEX)
        good = ((pred is None and gold is None) or
                (pred is not None and gold is not None and canon(pred) == canon(gold)))
        ok += int(good)
        if gold is None:
            inv_n += 1
            inv_ok += int(good)
        else:
            valid_n += 1
            valid_ok += int(good)
        t = by_tag.setdefault(row["tag"], [0, 0])
        t[0] += int(good)
        t[1] += 1
        if not good and len(errors) < 12:
            errors.append((row["tag"], sent, row["gold_desc"], describe(pred)))

    n = len(data)
    res = {
        "n": n,
        "freshly_generated": fresh,
        "overall": ok / n,
        "valid_exact": valid_ok / max(1, valid_n),
        "invalid_reject": inv_ok / max(1, inv_n),
        "valid_n": valid_n, "invalid_n": inv_n,
        "by_tag": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by_tag.items())},
        "accept_all_baseline": valid_n / n,
        "note": "标签来自手写验证器，不是独立人工真值。",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("frozen=%s n=%d (valid %d / invalid %d)" % (FROZEN, n, valid_n, inv_n))
    print("overall=%.4f valid=%.4f invalid=%.4f | accept-all=%.4f"
          % (res["overall"], res["valid_exact"], res["invalid_reject"],
             res["accept_all_baseline"]))
    print("-- by construction --")
    for k, v in res["by_tag"].items():
        print("  %-7s n=%3d acc=%.4f" % (k, v["n"], v["acc"]))
    if errors:
        print("-- first errors --")
        for tag, sent, g, p in errors:
            print("  [%s] %s | gold=%s | pred=%s" % (tag, sent, g, p))
    print("saved %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
