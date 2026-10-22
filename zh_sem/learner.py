# -*- coding: utf-8 -*-
"""离散打分器 v0.1：逐槽位否决（专家评审 A2）。

v0 的加性总分允许"合法槽位的正分抵掉非法槽位"，于是
  老师吃水 = 喝 迁移来的 role_liquid[PATIENT] +3 压过 吃 的 edible −4（假收）；
  鸡吃苹果 = −1 对 −1 平局被 reject 规则判拒（假拒）。
v0.1 改为：每个 (verb, role, filler) 先算槽位分，任一槽位 < 0 ⇒ 该候选无效；
只有全部槽位 ≥ 0 的候选才参与排序。一个坏槽位无法被别的槽位补偿。
"""
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lexicon import PROP_KEYS, entity_info  # noqa: E402
from parser import parse, tokenize  # noqa: E402
from semantics import canon  # noqa: E402


def slot_feats(verb, role, head, lex, mode="full"):
    fs = [("verb_role", verb, role)]
    if mode == "position_only":
        return [("role", role)]
    info = entity_info(lex, head)
    if info is None:
        return fs
    cls = info.get("cls")
    props = info.get("props")
    if props is None:
        return fs
    fs.append(("verb_role_class", verb, role, cls))
    fs.append(("role_class", role, cls))
    for k in PROP_KEYS:
        v = props[k]
        fs.append(("verb_role_%s" % k, verb, role, v))
        fs.append(("role_%s" % k, role, v))
    return fs


def cand_slots(m):
    return [(m["pred"], role, head) for role, head in m["roles"].items()]


def cand_feats(m, lex, mode="full"):
    fs = []
    for verb, role, head in cand_slots(m):
        fs += slot_feats(verb, role, head, lex, mode)
    if not m["roles"]:
        fs += [("entity",), ("entity_head", m["pred"])]
    return fs


def slot_score(w, verb, role, head, lex, mode="full"):
    return sum(w.get(f, 0) for f in slot_feats(verb, role, head, lex, mode))


def slot_valid(w, verb, role, head, lex, mode="full"):
    """逐属性否决，带回退；类约束只在动词专属证据下生效。"""
    if mode == "position_only":
        return True
    info = entity_info(lex, head)
    if info is None:
        return True
    cls = info.get("cls")
    # 类门：只在 (verb, role, cls) 有训练证据时否决（未见类不误杀）
    if w.get(("verb_role_class", verb, role, cls), 0) < 0:
        return False
    props = info.get("props")
    if props is None:
        return True
    for k in PROP_KEYS:
        v = props[k]
        wv = w.get(("verb_role_%s" % k, verb, role, v), 0)
        if wv != 0:
            gate = wv
        else:
            gate = w.get(("role_%s" % k, role, v), 0)
        if gate < 0:
            return False
    return True


def candidate_valid(w, m, lex, mode="full"):
    return all(slot_valid(w, v, r, h, lex, mode) for v, r, h in cand_slots(m))


def slot_scores(w, m, lex, mode="full"):
    return [slot_score(w, v, r, h, lex, mode) for v, r, h in cand_slots(m)]


def predict(w, cands, lex, mode="full"):
    """返回 (最优候选 或 None, 其特征)。None = 拒绝。

    有效性：候选的每一个槽位分都必须 >= 0。没有槽位的 entity 候选天然有效。
    """
    best = None
    best_s = None
    best_fs = None
    for m in cands:
        slots = cand_slots(m)
        if any(not slot_valid(w, v, r, h, lex, mode) for v, r, h in slots):
            continue                      # 逐属性/逐槽位否决
        total = sum(slot_score(w, v, r, h, lex, mode) for v, r, h in slots)
        if best_s is None or total > best_s:
            best, best_s, best_fs = m, total, cand_feats(m, lex, mode)
    return best, best_fs


def update(w, pos_fs, neg_fs):
    for f in pos_fs:
        w[f] += 1
    for f in neg_fs:
        w[f] -= 1


def _target(cands, gold):
    g = canon(gold)
    for m in cands:
        if canon(m) == g:
            return m
    return None


def train(data, lex, mode="full", epochs=40, seed=0):
    w = defaultdict(int)
    rng = random.Random(seed)
    for _ in range(epochs):
        items = list(data)
        rng.shuffle(items)
        for sent, gold in items:
            cands = parse(sent)
            pred, pf = predict(w, cands, lex, mode)
            if gold is None:
                # 负例：惩罚**所有**当前被接受的候选，否则另一个义项会漏学
                for m in cands:
                    if candidate_valid(w, m, lex, mode):
                        update(w, [], cand_feats(m, lex, mode))
            else:
                target = _target(cands, gold)
                if target is None:
                    raise AssertionError("parser 没产出 gold 候选: %r" % sent)
                tf = cand_feats(target, lex, mode)
                if pred is None:                     # 假拒绝
                    update(w, tf, [])
                elif canon(pred) != canon(gold):     # 选错候选
                    update(w, tf, pf)
    return w


def train_acc(w, data, lex, mode="full"):
    ok = 0
    for sent, gold in data:
        pred, _ = predict(w, parse(sent), lex, mode)
        good = (pred is None and gold is None) or (
            pred is not None and gold is not None and canon(pred) == canon(gold))
        ok += int(good)
    return ok, len(data)


def evaluate(w, data, lex, mode="full", verbose=False):
    stats = {"valid_ok": 0, "valid_n": 0, "invalid_ok": 0, "invalid_n": 0,
             "items": [], "errors": []}
    from semantics import describe
    for sent, gold in data:
        pred, _ = predict(w, parse(sent), lex, mode)
        if gold is None:
            stats["invalid_n"] += 1
            good = pred is None
            stats["invalid_ok"] += int(good)
        else:
            stats["valid_n"] += 1
            good = pred is not None and canon(pred) == canon(gold)
            stats["valid_ok"] += int(good)
        stats["items"].append({
            "sent": sent,
            "gold": describe(gold),
            "pred": describe(pred),
            "ok": bool(good),
        })
        if not good and verbose:
            stats["errors"].append((sent, describe(gold), describe(pred)))
    return stats


def ratio(stats, key):
    n = stats["valid_n"] if key == "valid" else stats["invalid_n"]
    return (stats["valid_ok"] if key == "valid" else stats["invalid_ok"]) / max(1, n)


def randomize_props(lex, seed=0):
    """负向对照：四个属性列在名词之间**独立**置换。

    保留每个属性的边际分布，破坏"哪个词有哪些属性"的相关结构。
    这才是真正的语义先验对照；v0 的 random_class 只改类名、gold 不变，
    测出来的是标签矛盾，不是语义先验。
    """
    nouns = [f for f, i in lex.items() if i["pos"] in ("N", "PRON")]
    rng = random.Random(seed)
    cols = {}
    for k in PROP_KEYS:
        vals = [lex[f]["props"][k] for f in nouns]
        rng.shuffle(vals)
        cols[k] = dict(zip(nouns, vals))
    out = {f: dict(i) for f, i in lex.items()}
    changed = 0
    for f in nouns:
        old = tuple(lex[f]["props"][k] for k in PROP_KEYS)
        new = {k: cols[k][f] for k in PROP_KEYS}
        out[f]["props"] = new
        if tuple(new[k] for k in PROP_KEYS) != old:
            changed += 1
    return out, changed


def token_forms(sent):
    return tokenize(sent)
