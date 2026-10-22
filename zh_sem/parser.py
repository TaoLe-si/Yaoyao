# -*- coding: utf-8 -*-
"""非 transformer 的显式组合解析器 v0.3。

新增构造（专家共识：先补结构覆盖）：
  - 并列：NP 和 NP -> 复合 NP
  - 疑问：S 吗 -> S[Q]
  - 否定形容词：不 + AP -> AP[NEG]
  - 数量短语：数词 + 量词 + N -> NP[QUANT]

仍然没有任何矩阵乘法、注意力或梯度；每条构造都是显式规则。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lexicon import ENTRIES, VERB_FRAME, VOPT   # noqa: E402
from semantics import canon, entity, event, state  # noqa: E402

_FORMS = sorted(ENTRIES, key=len, reverse=True)
_MAXLEN = max(len(f) for f in _FORMS)


def tokenize(sent):
    out = []
    i = 0
    while i < len(sent):
        matched = None
        for L in range(min(_MAXLEN, len(sent) - i), 0, -1):
            w = sent[i:i + L]
            if w in ENTRIES:
                matched = w
                break
        if matched is None:
            raise ValueError("OOV %r at %d in %r" % (sent[i:i + 1], i, sent))
        out.append(matched)
        i += len(matched)
    return out


def _copy_vp(vp):
    return {
        "verb": vp["verb"],
        "roles": dict(vp["roles"]),
        "mods": dict(vp["mods"]),
        "construction": vp.get("construction", "vp"),
    }


def _copy_vp2(vp, **mods):
    nv = _copy_vp(vp)
    nv["mods"].update({k: v for k, v in mods.items() if v})
    return nv


def _with_mods(m, **mods):
    n = dict(m)
    n["roles"] = dict(m["roles"])
    n["mods"] = dict(m["mods"])
    n["mods"].update({k: v for k, v in mods.items() if v})
    return n


def _fill_object(vp, np):
    if "verb" not in vp:
        return None
    for role in VERB_FRAME[vp["verb"]][1:]:
        if role not in vp["roles"]:
            nv = _copy_vp(vp)
            nv["roles"][role] = np["pred"]
            return nv
    return None


def _close_s(subj, vp):
    if "verb" not in vp:
        return None
    frame = VERB_FRAME[vp["verb"]]
    opt = VOPT[vp["verb"]]
    subj_role = frame[0]
    if subj_role in vp["roles"]:
        return None
    roles = dict(vp["roles"])
    roles[subj_role] = subj["pred"]
    if any(r not in roles for r in frame if r not in opt):
        return None
    return event(vp["verb"], roles, construction="SVO", **vp["mods"])


def _combine(L, R, add):
    # R1: NP + VP -> S
    for np in L.get("NP", []):
        for vp in R.get("VP", []):
            m = _close_s(np, vp)
            if m:
                add("S", m)

    # R2: NP + AP -> 形容词谓语句（含否定）
    for np in L.get("NP", []):
        for ap in R.get("AP", []):
            add("S", state(ap["adj"], {"THEME": np["pred"]},
                           DEGREE=ap.get("degree", False),
                           NEG=ap.get("neg", False),
                           construction="adj_state"))

    # R3: NP + PP(在) -> 在字句
    for np in L.get("NP", []):
        for pp in R.get("PP", []):
            if pp["prep"] == "在":
                add("S", state("在", {"THEME": np["pred"], "LOC": pp["obj"]["pred"]},
                               construction="loc_state"))

    # R4: ADV(不/今天) + VP -> VP
    for adv in L.get("ADV", []):
        if adv["kind"] in ("NEG", "TIME"):
            for vp in R.get("VP", []):
                add("VP", _copy_vp2(vp, **{adv["kind"]: True}))

    # R5: ADV(很) + AP -> AP
    for adv in L.get("ADV", []):
        if adv["kind"] == "DEGREE":
            for ap in R.get("AP", []):
                add("AP", {"adj": ap["adj"], "degree": True,
                           "neg": ap.get("neg", False)})

    # R5b: ADV(不) + AP -> AP（否定形容词）
    for adv in L.get("ADV", []):
        if adv["kind"] == "NEG":
            for ap in R.get("AP", []):
                add("AP", {"adj": ap["adj"], "degree": ap.get("degree", False),
                           "neg": True})

    # R6: VP + NP -> VP（动宾）
    for vp in L.get("VP", []):
        for np in R.get("NP", []):
            nv = _fill_object(vp, np)
            if nv:
                add("VP", nv)

    # R7: VP + PART(了) -> VP
    for vp in L.get("VP", []):
        for pt in R.get("PART", []):
            if pt["kind"] == "PERF":
                add("VP", _copy_vp2(vp, PERF=True))

    # R8: P + NP -> PP / PPBA / PPBEI
    for p in L.get("P", []):
        for np in R.get("NP", []):
            if p["prep"] == "在":
                add("PP", {"prep": "在", "obj": np})
            elif p["prep"] == "把":
                add("PPBA", {"obj": np})
            elif p["prep"] == "被":
                add("PPBEI", {"obj": np})

    # R9: PP(在) + VP -> VP（地点状语）
    for pp in L.get("PP", []):
        if pp["prep"] == "在":
            for vp in R.get("VP", []):
                add("VP", _copy_vp2(vp, LOC=pp["obj"]["pred"]))

    # R10: NP + PPBA -> NPBA
    for np in L.get("NP", []):
        for pp in R.get("PPBA", []):
            add("NPBA", {"agent": np, "patient": pp["obj"]})

    # R11: NP + PPBEI -> NPBEI
    for np in L.get("NP", []):
        for pp in R.get("PPBEI", []):
            add("NPBEI", {"patient": np, "agent": pp["obj"]})

    # R12: NPBA + VP -> S（把字句）
    for n in L.get("NPBA", []):
        for vp in R.get("VP", []):
            if "verb" not in vp:
                continue
            if "AGENT" in vp["roles"] or "PATIENT" in vp["roles"]:
                continue
            roles = dict(vp["roles"])
            roles["AGENT"] = n["agent"]["pred"]
            roles["PATIENT"] = n["patient"]["pred"]
            if all(r in roles for r in VERB_FRAME[vp["verb"]] if r not in VOPT[vp["verb"]]):
                add("S", event(vp["verb"], roles, construction="BA", BA=True, **vp["mods"]))

    # R13: NPBEI + VP -> S（被字句）
    for n in L.get("NPBEI", []):
        for vp in R.get("VP", []):
            if "verb" not in vp:
                continue
            if "AGENT" in vp["roles"] or "PATIENT" in vp["roles"]:
                continue
            roles = dict(vp["roles"])
            roles["PATIENT"] = n["patient"]["pred"]
            roles["AGENT"] = n["agent"]["pred"]
            if all(r in roles for r in VERB_FRAME[vp["verb"]] if r not in VOPT[vp["verb"]]):
                add("S", event(vp["verb"], roles, construction="BEI", BEI=True, **vp["mods"]))

    # R14: A + PART(的) -> APMOD
    for ap in L.get("AP", []):
        for pt in R.get("PART", []):
            if pt["kind"] == "MOD":
                add("APMOD", {"adj": ap["adj"]})

    # R15: APMOD + N -> NP（定中）
    for am in L.get("APMOD", []):
        for np in R.get("NP", []):
            add("NP", entity(np["pred"], base=np.get("base", np["pred"]),
                             ATTR=am["adj"]))

    # R16: CONJ + NP -> NP_TAIL
    for _ in L.get("CONJ", []):
        for np in R.get("NP", []):
            add("NP_TAIL", {"head": np["pred"]})

    # R17: NP + NP_TAIL -> 复合 NP（并列）
    for np in L.get("NP", []):
        for tail in R.get("NP_TAIL", []):
            add("NP", entity(np["pred"] + "+" + tail["head"], base=np["pred"] + "+" + tail["head"]))

    # R18: S + PART(吗) -> S[Q]
    for m in L.get("S", []):
        for pt in R.get("PART", []):
            if pt["kind"] == "Q":
                add("S", _with_mods(m, Q=True))

    # R19: NUM + CL -> CLP
    for num in L.get("NUM", []):
        for cl in R.get("CL", []):
            add("CLP", {"num": num["num"], "cl": cl["cl"]})

    # R20: CLP + N -> NP（数量短语）
    for q in L.get("CLP", []):
        for np in R.get("NP", []):
            base = np.get("base", np["pred"])
            add("NP", entity(q["num"] + q["cl"] + base, base=base,
                             QUANT=q["num"] + q["cl"]))


def parse(sent):
    forms = tokenize(sent)
    n = len(forms)
    chart = {}

    def add(i, j, cat, m):
        chart.setdefault((i, j), {}).setdefault(cat, []).append(m)

    for i, form in enumerate(forms):
        for e in ENTRIES[form]:
            pos = e["pos"]
            if pos in ("N", "PRON"):
                add(i, i + 1, "NP", entity(form))
            elif pos == "V":
                add(i, i + 1, "VP", {"verb": e["sense"], "roles": {}, "mods": {},
                                     "construction": "vp"})
            elif pos == "A":
                add(i, i + 1, "AP", {"adj": form, "degree": False, "neg": False})
            elif pos == "ADV":
                add(i, i + 1, "ADV", {"adv": form, "kind": e["kind"]})
            elif pos == "PART":
                add(i, i + 1, "PART", {"part": form, "kind": e["kind"]})
            elif pos == "P":
                add(i, i + 1, "P", {"prep": form})
            elif pos == "CONJ":
                add(i, i + 1, "CONJ", {"conj": form})
            elif pos == "NUM":
                add(i, i + 1, "NUM", {"num": form})
            elif pos == "CL":
                add(i, i + 1, "CL", {"cl": form})

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            for k in range(i + 1, j):
                L = chart.get((i, k))
                R = chart.get((k, j))
                if not L or not R:
                    continue
                _combine(L, R, lambda c, m, ii=i, jj=j: add(ii, jj, c, m))

    out = chart.get((0, n), {})
    cands = list(out.get("S", []))
    if not cands:
        for np in out.get("NP", []):
            if np["mods"].get("ATTR"):
                cands.append(np)

    seen = set()
    uniq = []
    for m in cands:
        c = canon(m)
        if c not in seen:
            seen.add(c)
            uniq.append(m)
    return uniq
