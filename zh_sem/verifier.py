# -*- coding: utf-8 -*-
"""手写验证器 v0.2：按义项给出角色允许的属性/词类。

`{"cls": "READABLE"}` 表示按词类约束（看/read 的 PATIENT 必须是可读物）；
其余键按正交属性约束。验证器等价于逐句人工反馈，同时是 RSI 的上限。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from parser import parse  # noqa: E402

ROLE_REQ = {
    ("吃", "AGENT"): {"animate": True},
    ("吃", "PATIENT"): {"edible": True},
    ("喝", "AGENT"): {"animate": True},
    ("喝", "PATIENT"): {"liquid": True},
    ("跑", "AGENT"): {"animate": True},
    ("看/read", "AGENT"): {"animate": True},
    ("看/read", "PATIENT"): {"cls": "READABLE"},
    ("看/visit", "AGENT"): {"animate": True},
    ("看/visit", "PATIENT"): {"place": True},
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


def verify(m, lex):
    if m is None:
        return False
    for role, head in m["roles"].items():
        req = ROLE_REQ.get((m["pred"], role))
        if not req:
            continue
        info = lex[head]
        props = info.get("props")
        if props is None:
            return False
        for k, v in req.items():
            if k == "cls":
                if info.get("cls") != v:
                    return False
            elif props.get(k) != v:
                return False
    return True


def label_sentence(sent, lex):
    for m in parse(sent):
        if verify(m, lex):
            return m
    return None
