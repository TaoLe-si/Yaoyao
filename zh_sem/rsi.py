# -*- coding: utf-8 -*-
"""RSI v0.1：验证器标签 vs 打乱标签 vs 不 RSI（专家评审 A6）。

v0 的问题：只跑一组，没有对照；验证器与 held-out 同源；P5"三轮不下降"
连常数模型都能通过。v0.1 改为同 pool、同 chunk、同 seed、同轮数的三组：

  A: 验证器伪标签
  B: 同 chunk 内把"接受/拒绝"模式随机置换（正负数量不变），被接受句用首候选
  C: 不做 RSI

评测集：pool 前 200 条冻结（与训练/held-out 不重叠），标签由验证器给出——
所以读数只能叫"与验证器一致率"，不能叫"语义正确率"。
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import HELDOUT, TRAIN  # noqa: E402
from learner import evaluate, train  # noqa: E402
from lexicon import LEX, VERB_FRAME  # noqa: E402
from parser import parse  # noqa: E402
from semantics import canon  # noqa: E402
from verifier import label_sentence  # noqa: E402

OUT = os.path.join(HERE, "rsi_results.json")


def build_pool(limit=900, seed=0):
    nouns = [f for f, i in LEX.items() if i["pos"] in ("N", "PRON")]
    verbs = [f for f, i in LEX.items() if i["pos"] == "V"]
    adjs = [f for f, i in LEX.items() if i["pos"] == "A"]
    fixed = {s for s, _ in TRAIN} | {s for s, _ in HELDOUT}
    raw = []
    for v in verbs:
        frame = LEX[v]["frame"]
        if len(frame) == 1:
            raw += ["%s%s" % (n, v) for n in nouns]
        elif len(frame) == 2:
            raw += ["%s%s%s" % (n1, v, n2) for n1 in nouns for n2 in nouns]
        elif len(frame) == 3:
            raw += ["%s%s%s%s" % (n1, v, n2, n3)
                    for n1 in nouns for n2 in nouns for n3 in nouns]
    raw += ["%s在%s" % (n, p) for n in nouns for p in nouns]
    raw += ["%s很%s" % (n, a) for n in nouns for a in adjs]
    raw += ["%s的%s" % (a, n) for a in adjs for n in nouns]
    rng = random.Random(seed)
    rng.shuffle(raw)
    out = []
    for s in raw:
        if s in fixed or s in out:
            continue
        try:
            if not parse(s):
                continue
        except ValueError:
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def signatures(sent):
    g = label_sentence(sent, LEX)
    cands = [g] if g is not None else parse(sent)
    sig = set()
    for m in cands:
        for role, head in m["roles"].items():
            sig.add((m["pred"], role, LEX[head].get("cls")))
    return sig


def shuffled_teacher(chunk, seed):
    """B 组：保留正负数量，随机置换接受/拒绝模式。"""
    labels = [label_sentence(s, LEX) for s in chunk]
    flags = [g is not None for g in labels]
    rng = random.Random(seed)
    rng.shuffle(flags)
    out = []
    for s, flag in zip(chunk, flags):
        if flag:
            cands = parse(s)
            out.append((s, cands[0] if cands else None))
        else:
            out.append((s, None))
    return out


def mcnemar(a_items, b_items):
    a_only = sum(1 for x, y in zip(a_items, b_items) if x["ok"] and not y["ok"])
    b_only = sum(1 for x, y in zip(a_items, b_items) if not x["ok"] and y["ok"])
    n = a_only + b_only
    if n == 0:
        return 1.0, a_only, b_only
    k = min(a_only, b_only)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p), a_only, b_only


def run_group(name, eval_set, unlabeled, chunks, teacher):
    data = list(TRAIN)
    rows = []
    for rnd, chunk in enumerate(chunks, 1):
        data += teacher(chunk, rnd) if teacher else []
        w = train(data, LEX, epochs=20, seed=0)   # 固定 seed=0，不和数据变化混淆
        st = evaluate(w, eval_set, LEX)
        n = len(eval_set)
        acc = (st["valid_ok"] + st["invalid_ok"]) / n
        rows.append({"round": rnd, "n_train": len(data), "accuracy": acc,
                     "valid_exact": st["valid_ok"] / max(1, st["valid_n"]),
                     "invalid_reject": st["invalid_ok"] / max(1, st["invalid_n"]),
                     "items": st["items"]})
        print("  %s round %d: train=%d acc=%.4f" % (name, rnd, len(data), acc))
    return rows


def main():
    pool = build_pool()
    eval_pool = pool[:200]
    unlabeled = pool[200:]
    eval_set = [(s, label_sentence(s, LEX)) for s in eval_pool]
    chunks = [unlabeled[i:i + 60] for i in range(0, 180, 60)]

    eval_sigs = set().union(*(signatures(s) for s in eval_pool)) if eval_pool else set()
    chunk_sigs = set().union(*(signatures(s) for s in sum(chunks, []))) if chunks else set()
    overlap = len(eval_sigs & chunk_sigs)
    print("pool=%d eval=%d unlabeled=%d; 签名重叠=%d/%d"
          % (len(pool), len(eval_pool), len(unlabeled), overlap, len(eval_sigs)))

    print("A: verifier labels")
    a_rows = run_group("A", eval_set, unlabeled, chunks, lambda c, r: [(s, label_sentence(s, LEX)) for s in c])
    print("B: shuffled accept/reject pattern")
    b_rows = run_group("B", eval_set, unlabeled, chunks, shuffled_teacher)
    print("C: no RSI")
    w0 = train(TRAIN, LEX, epochs=20, seed=0)
    st0 = evaluate(w0, eval_set, LEX)
    c_acc = (st0["valid_ok"] + st0["invalid_ok"]) / len(eval_set)
    print("  C: train=%d acc=%.4f" % (len(TRAIN), c_acc))

    p, a_only, b_only = mcnemar(a_rows[-1]["items"], b_rows[-1]["items"])
    a_acc = a_rows[-1]["accuracy"]
    b_acc = b_rows[-1]["accuracy"]
    res = {
        "n_eval": len(eval_set),
        "signature_overlap": {"count": overlap, "n_eval_sigs": len(eval_sigs)},
        "A": [{k: v for k, v in r.items() if k != "items"} for r in a_rows],
        "B": [{k: v for k, v in r.items() if k != "items"} for r in b_rows],
        "C": {"accuracy": c_acc},
        "mcnemar_AB": {"p": p, "A_only_correct": a_only, "B_only_correct": b_only},
        "P6_rsi_effective": (a_acc - b_acc) >= 0.10 and p < 0.05,
        "note": "评测标签来自手写验证器；这里的读数只能叫『与验证器一致率』，"
                "验证器是模型的上限，不是独立真值。",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\nA=%.4f B=%.4f C=%.4f | A-B=%+.4f McNemar p=%.4f (A_only=%d B_only=%d)"
          % (a_acc, b_acc, c_acc, a_acc - b_acc, p, a_only, b_only))
    print("P6 RSI 有效（A-B>=10pp 且 p<0.05）: %s" % res["P6_rsi_effective"])
    print("saved %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
