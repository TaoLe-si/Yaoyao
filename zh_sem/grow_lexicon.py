# -*- coding: utf-8 -*-
"""从语料自动抽取候选词条，并用主动反馈筛选进词典。

流程：
  1. 字符 n-gram（2–4）频次 → 候选词（去掉已知词、功能词边界）；
  2. 教师表（手写，代表"人给的词义反馈"）决定候选是否收、词性/属性/框架；
  3. 收词后生成该词的句子池，教师按角色约束打标；
  4. 主动挑模型判错的句子要标签，重训，直到 ≥0.9 或预算耗尽；
  5. 对比收词前后在"成长测试集"上的准确率。

★教师表是人工反馈的替身；这里的读数仍不是独立人工 held-out。
"""
import json
import os
import random
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lexicon as LEXMOD  # noqa: E402
from data import TRAIN  # noqa: E402
from learner import predict, train  # noqa: E402
from lexicon import ENTRIES, LEX, entity_info  # noqa: E402
from parser import parse, tokenize  # noqa: E402
from semantics import canon, describe  # noqa: E402
from verifier import ROLE_REQ  # noqa: E402

CORPUS = os.path.join(HERE, "corpus_zh.txt")
OUT = os.path.join(HERE, "growth_results.json")
MAX_LABELS = 6

STOP = set("我你他和不了的很在一个本只吗把被")

TEACHER = {
    "菠萝": dict(pos="N", cls="FRUIT", props=dict(animate=False, edible=True, liquid=False,
                                                place=False, readable=False, sport=False)),
    "榴莲": dict(pos="N", cls="FRUIT", props=dict(animate=False, edible=True, liquid=False,
                                                place=False, readable=False, sport=False)),
    "草莓": dict(pos="N", cls="FRUIT", props=dict(animate=False, edible=True, liquid=False,
                                                place=False, readable=False, sport=False)),
    "可乐": dict(pos="N", cls="LIQUID", props=dict(animate=False, edible=False, liquid=True,
                                                 place=False, readable=False, sport=False)),
    "小说": dict(pos="N", cls="READABLE", props=dict(animate=False, edible=False, liquid=False,
                                                   place=False, readable=True, sport=False)),
    "游泳": dict(pos="V", sense="游泳", frame=["AGENT"], opt=[]),
    "唱歌": dict(pos="V", sense="唱歌", frame=["AGENT"], opt=[]),
    "跳舞": dict(pos="V", sense="跳舞", frame=["AGENT"], opt=[]),
    "跑步": dict(pos="V", sense="跑步", frame=["AGENT"], opt=[]),
    "阅读": dict(pos="V", sense="阅读", frame=["AGENT", "PATIENT"], opt=["PATIENT"]),
    "穿": dict(pos="V", sense="穿", frame=["AGENT", "PATIENT"], opt=[]),
}

# 教师对个别词给出的专属约束（新维度：readable / clothing）
TEACHER_REQ = {
    ("阅读", "PATIENT"): {"readable": True},
    ("穿", "PATIENT"): {"clothing": True},
}


def extract_candidates(min_count=5, limit=20):
    text = "".join(open(CORPUS, encoding="utf-8").read().split())
    grams = Counter()
    for n in (1, 2, 3, 4):
        for i in range(len(text) - n + 1):
            g = text[i:i + n]
            if any(not ("\u4e00" <= c <= "\u9fff") for c in g):
                continue
            if g in ENTRIES:
                continue
            # 单字只收教师认识的（否则会被 妈/衣/服 这类高频单字淹没）
            if n == 1 and g not in TEACHER:
                continue
            if g[0] in STOP or g[-1] in STOP:
                continue
            grams[g] += 1
    ranked = [(g, c) for g, c in grams.items() if c >= min_count]
    ranked.sort(key=lambda x: (-x[1], -len(x[0])))
    return ranked[:limit]


def _norm_props(p):
    from lexicon import PROP_KEYS
    return {k: bool(p.get(k, False)) for k in PROP_KEYS}


def _info(head, override):
    if head == override["word"]:
        info = dict(override)
        if "props" in info:
            info["props"] = _norm_props(info["props"])
        return info
    return entity_info(LEX, head)


def teacher_verify(m, override):
    if m is None:
        return False
    for role, head in m["roles"].items():
        req = ROLE_REQ.get((m["pred"], role))
        if not req:
            continue
        info = _info(head, override)
        if info is None:
            return False
        props = info.get("props")
        for k, v in req.items():
            if k == "cls":
                if info.get("cls") != v:
                    return False
            elif k == "cls_not":
                if info.get("cls") == v:
                    return False
            elif props is None or props.get(k) != v:
                return False
    return True


def teacher_label(sent, override):
    for m in parse(sent, strict=False):
        if teacher_verify(m, override):
            return m
    return None


def install(word, entry):
    if entry["pos"] == "N":
        props = _norm_props(entry["props"])
        e = dict(pos="N", cls=entry["cls"], props=props)
        ENTRIES[word] = [e]
        LEX[word] = e
        LEXMOD.NOUN_PROPS[word] = props
    else:
        e = dict(pos="V", sense=entry["sense"],
                 frame=list(entry["frame"]), opt=list(entry.get("opt", [])))
        ENTRIES[word] = [e]
        LEX[word] = e
        # parser 在导入时绑定了同一个 VERB_FRAME/VOPT 对象 ⇒ 原地更新才可见
        LEXMOD.VERB_FRAME[e["sense"]] = e["frame"]
        LEXMOD.VOPT[e["sense"]] = e["opt"]
        for role in e["frame"]:
            if role in ("AGENT", "EXPERIENCER", "POSSESSOR", "RECIPIENT"):
                ROLE_REQ.setdefault((e["sense"], role), {"animate": True})
            elif role == "GOAL":
                ROLE_REQ.setdefault((e["sense"], role), {"place": True})
        for key, req in TEACHER_REQ.items():
            if key[0] == e["sense"]:
                ROLE_REQ[key] = dict(req)


def build_pool(word, entry):
    if entry["pos"] == "N":
        tmpl = ["我吃%s", "我喝%s", "老师看%s", "我买%s", "我给%s书",
                "妈妈给%s苹果", "哥哥吃%s", "妈妈买%s"]
    else:
        tmpl = ["我%s", "妈妈%s", "老师%s", "哥哥%s", "弟弟%s",
                "妹妹%s", "爷爷%s", "奶奶%s", "石头%s", "桌子%s",
                "老师%s小说", "我%s报纸", "弟弟%s桌子", "妈妈%s衣服",
                "爷爷%s书", "妹妹%s杂志", "我%s苹果", "哥哥%s球",
                "爷爷%s衣服", "奶奶%s鞋", "我%s帽子", "妈妈%s苹果",
                "老师%s书", "哥哥%s桌子", "弟弟%s报纸", "妹妹%s小说",
                "姐姐%s杂志", "爸爸%s电脑", "学生%s手机", "工人%s电视"]
    pool = []
    for t in tmpl:
        s = t % word
        try:
            if not parse(s, strict=False):
                continue
        except ValueError:
            continue
        g = teacher_label(s, dict(entry, word=word))
        pool.append((s, g))
    return pool


def _same(pred, gold):
    return ((pred is None and gold is None) or
            (pred is not None and gold is not None and canon(pred) == canon(gold)))


def accuracy(w, items):
    ok = 0
    for s, g in items:
        cands = parse(s, strict=False)
        pred = None
        if cands:
            pred, _ = predict(w, cands, LEX)
        ok += int(_same(pred, g))
    return ok / max(1, len(items))


def main():
    rng = random.Random(0)
    proposals = extract_candidates()
    print("候选词（n-gram）:", ["%s:%d" % (g, c) for g, c in proposals[:14]])

    base = train(TRAIN, LEX, epochs=20, seed=0)
    rows = []
    growth_data = list(TRAIN)
    all_test = []
    for word, count in proposals:
        entry = TEACHER.get(word)
        if entry is None:
            rows.append({"word": word, "count": count, "accepted": False,
                         "reason": "教师未确认"})
            continue
        install(word, entry)
        pool = build_pool(word, entry)
        rng.shuffle(pool)
        test = pool[:4]
        query = pool[4:]
        before = accuracy(base, test)
        w = train(growth_data, LEX, epochs=15, seed=0)
        init = accuracy(w, test)
        labels = 0
        acc = init
        while labels < MAX_LABELS and acc < 0.9 and query:
            picked = None
            for s, g in query:
                cands = parse(s, strict=False)
                pred = None
                if cands:
                    pred, _ = predict(w, cands, LEX)
                if not _same(pred, g):
                    picked = (s, g)
                    break
            if picked is None:
                break
            query.remove(picked)
            growth_data.append(picked)
            labels += 1
            w = train(growth_data, LEX, epochs=15, seed=0)
            acc = accuracy(w, test)
        all_test += test
        rows.append({"word": word, "count": count, "accepted": True,
                     "pos": entry["pos"], "labels": labels,
                     "before_acc": before, "init_acc": init, "final_acc": acc})
        print("  %-3s count=%2d %s labels=%d before=%.2f init=%.2f final=%.2f"
              % (word, count, entry["pos"], labels, before, init, acc))

    final = train(growth_data, LEX, epochs=20, seed=0)
    after = accuracy(final, all_test)
    before_all = accuracy(base, all_test)
    labels_total = sum(r.get("labels", 0) for r in rows)
    res = {
        "proposals": [{"word": g, "count": c} for g, c in proposals],
        "rows": rows,
        "n_accepted": sum(1 for r in rows if r.get("accepted")),
        "labels_total": labels_total,
        "before_acc": before_all,
        "after_acc": after,
        "note": "教师表是人工反馈的替身；候选来自字符 n-gram 频次。",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("accepted=%d labels=%d | growth test before=%.3f after=%.3f"
          % (res["n_accepted"], labels_total, before_all, after))
    print("saved %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
