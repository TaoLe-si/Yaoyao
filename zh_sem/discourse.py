# -*- coding: utf-8 -*-
"""跨句指代 + 对话状态（规则式，非学习）。

规则：
  - 他 -> 最近的人类（cls == HUMAN）
  - 它 -> 最近的非人类（cls != HUMAN）
  - "最近" = 语境提及列表里从后往前第一个匹配项
  - 没有匹配先行词 => 该句解析失败（必须拒绝）

语境提及按**词序**收集（token 顺序），不是按句义 roles 的插入顺序。
代词解析发生在句义生成之后、验证之前：把角色里的代词换成先行词，再用验证器检查。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lexicon import LEX  # noqa: E402
from parser import parse, tokenize  # noqa: E402
from verifier import ROLE_REQ, verify  # noqa: E402

PRONOUNS = {
    "他": lambda info: bool(info) and info.get("cls") == "HUMAN",
    "它": lambda info: bool(info) and info.get("cls") != "HUMAN",
}


def _mention_ok(m, req):
    if not req:
        return True
    props = m.get("props") or {}
    for k, v in req.items():
        if k == "cls":
            if m.get("cls") != v:
                return False
        elif k == "cls_not":
            if m.get("cls") == v:
                return False
        elif props.get(k) != v:
            return False
    return True


def find_antecedent(pron, mentions, req=None, nearest=True):
    """最近匹配先行词；req 是谓词对该角色的选择限制（用于过滤不兼容的先行词）。"""
    seq = reversed(mentions) if nearest else iter(mentions)
    for m in seq:
        if PRONOUNS[pron](m) and _mention_ok(m, req):
            return m
    return None


def collect_mentions(sent, context, nearest=True):
    """按词序收集实体提及；代词就地解析。返回 (mentions, ok)。"""
    out = []
    for tok in tokenize(sent, strict=True):
        if tok in PRONOUNS:
            ant = find_antecedent(tok, context + out, nearest=nearest)
            if ant is None:
                return out, False
            out.append(ant)
        else:
            info = LEX.get(tok)
            if info is not None and info.get("props") is not None:
                out.append({"head": tok, "cls": info.get("cls"),
                            "props": info["props"]})
    return out, True


def _resolve_head(head, context, pred=None, role=None, nearest=True):
    if isinstance(head, dict):
        return _resolve_meaning(head, context, nearest=nearest)
    if head in PRONOUNS:
        req = ROLE_REQ.get((pred, role))
        ant = find_antecedent(head, context, req=req, nearest=nearest)
        return ant["head"] if ant else None
    return head


def _resolve_meaning(m, context, nearest=True):
    n = dict(m)
    n["roles"] = {}
    for role, head in m["roles"].items():
        h = _resolve_head(head, context, m["pred"], role, nearest=nearest)
        if h is None:
            return None
        n["roles"][role] = h
    if m.get("comps"):
        comps = {}
        for k, comp in m["comps"].items():
            c = _resolve_meaning(comp, context, nearest=nearest)
            if c is None:
                return None
            comps[k] = c
        n["comps"] = comps
    return n


def resolve_candidates(sent, context, nearest=True):
    """解析并解析指代；返回能成功绑定的候选（尚未过验证器）。"""
    out = []
    for m in parse(sent, strict=True):
        r = _resolve_meaning(m, context, nearest=nearest)
        if r is not None:
            out.append(r)
    return out


def process_sentence(sent, context, nearest=True):
    """返回句义（已绑定指代并通过验证）或 None。"""
    for r in resolve_candidates(sent, context, nearest=nearest):
        if verify(r, LEX):
            return r
    return None


def process_dialogue(sentences, nearest=True):
    """逐句处理；返回每句的句义（失败为 None），语境累积提及。"""
    ctx = []
    out = []
    for s in sentences:
        m = process_sentence(s, ctx, nearest=nearest)
        out.append(m)
        if m is None:
            break
        mentions, ok = collect_mentions(s, ctx, nearest=nearest)
        if not ok:
            out[-1] = None
            break
        ctx.extend(mentions)
    return out


def process_pair(s1, s2, nearest=True):
    return process_dialogue([s1, s2], nearest=nearest)
