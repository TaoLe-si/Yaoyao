# -*- coding: utf-8 -*-
"""手写验证器：每个动词/状态词的角色允许什么属性。

它等价于"逐句人工反馈"，同时是 RSI 的上限——模型只能逼近它。
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
    ("看", "AGENT"): {"animate": True},
    ("喜欢", "EXPERIENCER"): {"animate": True},
    ("去", "AGENT"): {"animate": True},
    ("去", "GOAL"): {"place": True},
    ("给", "AGENT"): {"animate": True},
    ("给", "RECIPIENT"): {"animate": True},
    ("有", "POSSESSOR"): {"animate": True},
    ("买", "AGENT"): {"animate": True},
    ("在", "LOC"): {"place": True},
}


def verify(m, lex):
    if m is None:
        return False
    for role, head in m["roles"].items():
        req = ROLE_REQ.get((m["pred"], role))
        if not req:
            continue
        props = lex[head].get("props")
        if props is None:
            return False
        for k, v in req.items():
            if props.get(k) != v:
                return False
    return True


def label_sentence(sent, lex):
    """验证器伪标签：任一候选通过 → 该候选；否则 REJECT。"""
    for m in parse(sent):
        if verify(m, lex):
            return m
    return None
