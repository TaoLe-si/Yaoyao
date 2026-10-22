# -*- coding: utf-8 -*-
"""手写验证器 v0.2：按义项给出角色允许的属性/词类。

`{"cls": "READABLE"}` 表示按词类约束（看/read 的 PATIENT 必须是可读物）；
其余键按正交属性约束。验证器等价于逐句人工反馈，同时是 RSI 的上限。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lexicon import entity_info  # noqa: E402
from parser import parse  # noqa: E402

ROLE_REQ = {
    ("吃", "AGENT"): {"animate": True},
    ("吃", "PATIENT"): {"edible": True},
    ("喝", "AGENT"): {"animate": True},
    ("喝", "PATIENT"): {"liquid": True},
    ("跑", "AGENT"): {"animate": True},
    ("看/read", "AGENT"): {"animate": True},
    ("看/read", "PATIENT"): {"readable": True},
    ("看/visit", "AGENT"): {"animate": True},
    ("看/visit", "PATIENT"): {"place": True},
    ("打/hit", "AGENT"): {"animate": True},
    ("打/hit", "PATIENT"): {"sport": False},
    ("打/play", "AGENT"): {"animate": True},
    ("打/play", "PATIENT"): {"sport": True},
    ("喜欢", "EXPERIENCER"): {"animate": True},
    ("去", "AGENT"): {"animate": True},
    ("去", "GOAL"): {"place": True},
    ("给", "AGENT"): {"animate": True},
    ("给", "RECIPIENT"): {"animate": True},
    ("有", "POSSESSOR"): {"animate": True},
    ("买", "AGENT"): {"animate": True},
    ("在", "LOC"): {"place": True},
    ("高兴", "THEME"): {"animate": True},
    ("忙", "THEME"): {"animate": True},
}

# 扩展词表：按角色名批量给通用约束（AGENT/EXPERIENCER/POSSESSOR/RECIPIENT animate，
# GOAL place），再对少数动词做专属覆盖。这是"手写语义"的可扩展写法，不是自动学习。
from lexicon import ENTRIES  # noqa: E402

for _es in ENTRIES.values():
    for _e in _es:
        if _e["pos"] != "V":
            continue
        for _role in _e["frame"]:
            if _role in ("AGENT", "EXPERIENCER", "POSSESSOR", "RECIPIENT"):
                ROLE_REQ.setdefault((_e["sense"], _role), {"animate": True})
            elif _role == "GOAL":
                ROLE_REQ.setdefault((_e["sense"], _role), {"place": True})

for _sense in ("写", "读"):
    ROLE_REQ.setdefault((_sense, "PATIENT"), {"readable": True})
for _adj in ("高兴", "忙", "聪明"):
    ROLE_REQ.setdefault((_adj, "THEME"), {"animate": True})


def verify(m, lex):
    if m is None:
        return False
    for role, head in m["roles"].items():
        req = ROLE_REQ.get((m["pred"], role))
        if not req:
            continue
        info = entity_info(lex, head)
        if info is None:
            return False
        props = info.get("props")
        if props is None:
            return False
        for k, v in req.items():
            if k == "cls":
                if info.get("cls") != v:
                    return False
            elif k == "cls_not":
                if info.get("cls") == v:
                    return False
            elif props.get(k) != v:
                return False
    return True


def label_sentence(sent, lex):
    for m in parse(sent):
        if verify(m, lex):
            return m
    return None


def violating_roles(m, lex):
    """反馈里的"原因"：列出违反 ROLE_REQ 的角色。

    训练负例只惩罚这些角色，避免把无关槽位也学成负证据。
    """
    bad = []
    for role, head in m["roles"].items():
        req = ROLE_REQ.get((m["pred"], role))
        if not req:
            continue
        info = entity_info(lex, head)
        if info is None:
            bad.append(role)
            continue
        props = info.get("props") or {}
        for k, v in req.items():
            if k == "cls":
                if info.get("cls") != v:
                    bad.append(role)
                    break
            elif k == "cls_not":
                if info.get("cls") == v:
                    bad.append(role)
                    break
            elif props.get(k) != v:
                bad.append(role)
                break
    return bad
