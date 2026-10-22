# -*- coding: utf-8 -*-
"""成本曲线：每学一个新动词需要多少条针对性反馈？

对每个新动词：
  1. 用验证器构造 8 正 + 8 负的测试集、12 正 + 12 负的查询池；
  2. 从 TRAIN（不含该词）冷启动；
  3. 主动挑模型当前判错的句子，向验证器要标签，重训；
  4. 测试集准确率 >= 0.9 或标签预算耗尽即停。

★标签与测试集都来自手写验证器；读数叫"与验证器一致率"。
"""
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import TRAIN  # noqa: E402
from learner import predict, train  # noqa: E402
from lexicon import LEX, entity_info  # noqa: E402
from parser import parse  # noqa: E402
from semantics import canon  # noqa: E402
from verifier import ROLE_REQ, label_sentence  # noqa: E402

OUT = os.path.join(HERE, "cost_curve.json")
MAX_LABELS = 12
NEW_VERBS = ["写", "读", "说", "听", "唱", "跳", "走", "来", "回", "睡", "站",
             "洗", "开", "关", "找", "等", "问", "教", "学", "做", "用",
             "送", "卖", "认识", "爱", "知道"]


def _matches(info, req):
    if not req:
        return True
    for k, v in req.items():
        if k == "cls":
            if info.get("cls") != v:
                return False
        elif k == "cls_not":
            if info.get("cls") == v:
                return False
        elif info.get("props", {}).get(k) != v:
            return False
    return True


def _sentence(word, forms):
    return "".join(forms[:1] + [word] + forms[1:])


def build_pools(word, nouns, rng):
    frame = LEX[word]["frame"]
    reqs = [ROLE_REQ.get((word, r), {}) for r in frame]
    sat, viol = [], []
    for r in reqs:
        sat.append([n for n in nouns if _matches(entity_info(LEX, n), r)])
        viol.append([n for n in nouns if not _matches(entity_info(LEX, n), r)])
    valid, invalid = [], []
    for _ in range(200):
        forms = [rng.choice(sat[i]) if sat[i] else rng.choice(nouns)
                 for i in range(len(frame))]
        valid.append(_sentence(word, forms))
    for i, r in enumerate(reqs):
        if not r or not viol[i]:
            continue
        for _ in range(40):
            forms = [rng.choice(sat[j]) if sat[j] else rng.choice(nouns)
                     for j in range(len(frame))]
            forms[i] = rng.choice(viol[i])
            invalid.append(_sentence(word, forms))
    fixed = {s for s, _ in TRAIN}
    out = {"valid": [], "invalid": []}
    for s in valid:
        if s in fixed or s in out["valid"]:
            continue
        try:
            if not parse(s):
                continue
        except ValueError:
            continue
        g = label_sentence(s, LEX)
        if g is not None:
            out["valid"].append((s, g))
    for s in invalid:
        if s in fixed or s in out["invalid"]:
            continue
        try:
            if not parse(s):
                continue
        except ValueError:
            continue
        if label_sentence(s, LEX) is None:
            out["invalid"].append((s, None))
    rng.shuffle(out["valid"])
    rng.shuffle(out["invalid"])
    return out


def accuracy(w, items):
    ok = 0
    for s, g in items:
        pred, _ = predict(w, parse(s), LEX)
        same = ((pred is None and g is None) or
                (pred is not None and g is not None and canon(pred) == canon(g)))
        ok += int(same)
    return ok / max(1, len(items))


def run_word(word, nouns, rng):
    pools = build_pools(word, nouns, rng)
    if len(pools["valid"]) < 4 or len(pools["invalid"]) < 4:
        return {"word": word, "note": "池不足", "n_valid": len(pools["valid"]),
                "n_invalid": len(pools["invalid"])}
    test = pools["valid"][:8] + pools["invalid"][:8]
    query = pools["valid"][8:20] + pools["invalid"][8:20]
    rng.shuffle(test)
    rng.shuffle(query)

    data = list(TRAIN)
    w = train(data, LEX, epochs=15, seed=0)
    init = accuracy(w, test)
    labels = 0
    acc = init
    while labels < MAX_LABELS and acc < 0.9 and query:
        picked = None
        for s, g in query:
            pred, _ = predict(w, parse(s), LEX)
            same = ((pred is None and g is None) or
                    (pred is not None and g is not None and canon(pred) == canon(g)))
            if not same:
                picked = (s, g)
                break
        if picked is None:
            break
        query.remove(picked)
        data.append(picked)
        labels += 1
        w = train(data, LEX, epochs=15, seed=0)
        acc = accuracy(w, test)
    return {"word": word, "labels": labels, "init_acc": init, "final_acc": acc,
            "reached_0.9": acc >= 0.9, "n_test": len(test)}


def main():
    nouns = [f for f, i in LEX.items() if i["pos"] in ("N", "PRON")]
    rng = random.Random(0)
    rows = [run_word(w, nouns, rng) for w in NEW_VERBS]
    done = [r for r in rows if "labels" in r]
    reached = [r for r in done if r["reached_0.9"]]
    summary = {
        "n_words": len(done),
        "n_reached_0.9": len(reached),
        "labels_mean": statistics.mean(r["labels"] for r in done) if done else None,
        "labels_median": statistics.median(r["labels"] for r in done) if done else None,
        "labels_max": max((r["labels"] for r in done), default=None),
        "labels_zero": sum(1 for r in done if r["labels"] == 0),
    }
    res = {"summary": summary, "rows": rows, "max_labels": MAX_LABELS,
           "note": "标签与测试集均来自手写验证器；读数是与验证器一致率。"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("words=%d reached>=0.9: %d" % (summary["n_words"], summary["n_reached_0.9"]))
    print("labels mean=%.2f median=%.1f max=%s zero=%d"
          % (summary["labels_mean"], summary["labels_median"],
             summary["labels_max"], summary["labels_zero"]))
    for r in rows:
        if "labels" in r:
            print("  %-3s labels=%2d init=%.2f final=%.2f %s"
                  % (r["word"], r["labels"], r["init_acc"], r["final_acc"],
                     "OK" if r["reached_0.9"] else "MISS"))
        else:
            print("  %-3s %s" % (r["word"], r.get("note", "")))
    print("saved %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
