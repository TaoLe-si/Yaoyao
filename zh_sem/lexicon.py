# -*- coding: utf-8 -*-
"""手写字词语义种子 v0.2。

v0.2 关键变化（专家评审 #1：候选泄漏 / 无歧义）：
  一个词形可以有**多个义项**。`ENTRIES[form]` 是义项列表。
  例：看 = 看/read（PATIENT 必须可读）与 看/visit（PATIENT 必须是地点）。
  两个义项句法框架相同 ⇒ 解析器产出**两个候选**，exact match 必须选对义项，
  不再等价于"接受/拒绝"。

字段：
  pos   : N / PRON / V / A / ADV / P / PART
  cls   : 粗粒度语义类（只作一个特征）
  props : 正交属性 animate / edible / liquid / place（逐词手写）
  sense : 仅动词；义项 id（同时是句义帧的 pred）
  frame : 仅动词；有序论元框架，第一项是主语角色
  opt   : 仅动词；可省略的论元
"""

NOUN_PROPS = {
    "我":    dict(animate=True,  edible=False, liquid=False, place=False),
    "你":    dict(animate=True,  edible=False, liquid=False, place=False),
    "他":    dict(animate=True,  edible=False, liquid=False, place=False),
    "小明":  dict(animate=True,  edible=False, liquid=False, place=False),
    "妈妈":  dict(animate=True,  edible=False, liquid=False, place=False),
    "老师":  dict(animate=True,  edible=False, liquid=False, place=False),
    "猫":    dict(animate=True,  edible=False, liquid=False, place=False),
    "狗":    dict(animate=True,  edible=False, liquid=False, place=False),
    "鸟":    dict(animate=True,  edible=False, liquid=False, place=False),
    "鸡":    dict(animate=True,  edible=True,  liquid=False, place=False),
    "鱼":    dict(animate=True,  edible=True,  liquid=False, place=False),
    "苹果":  dict(animate=False, edible=True,  liquid=False, place=False),
    "饭":    dict(animate=False, edible=True,  liquid=False, place=False),
    "水":    dict(animate=False, edible=False, liquid=True,  place=False),
    "书":    dict(animate=False, edible=False, liquid=False, place=False),
    "桌子":  dict(animate=False, edible=False, liquid=False, place=False),
    "石头":  dict(animate=False, edible=False, liquid=False, place=False),
    "学校":  dict(animate=False, edible=False, liquid=False, place=True),
    "家":    dict(animate=False, edible=False, liquid=False, place=True),
    "花":    dict(animate=False, edible=False, liquid=False, place=False),
}


def N(form, cls):
    return dict(pos="N", cls=cls, props=NOUN_PROPS[form])


def PRON(form):
    return dict(pos="PRON", cls="HUMAN", props=NOUN_PROPS[form])


def V(form, sense, frame, opt=None):
    return dict(pos="V", sense=sense, frame=list(frame), opt=list(opt or []))


def A():
    return dict(pos="A")


def ADV(kind):
    return dict(pos="ADV", kind=kind)


def PART(kind):
    return dict(pos="PART", kind=kind)


def P():
    return dict(pos="P")


ENTRIES = {
    # ---- 代词 ----
    "我": [PRON("我")], "你": [PRON("你")], "他": [PRON("他")],
    # ---- 名词 ----
    "小明": [N("小明", "HUMAN")],
    "妈妈": [N("妈妈", "HUMAN")],
    "老师": [N("老师", "HUMAN")],
    "猫": [N("猫", "ANIMAL")],
    "狗": [N("狗", "ANIMAL")],
    "鸟": [N("鸟", "ANIMAL")],
    "鸡": [N("鸡", "ANIMAL")],
    "鱼": [N("鱼", "ANIMAL")],
    "苹果": [N("苹果", "FRUIT")],
    "饭": [N("饭", "FOOD")],
    "水": [N("水", "LIQUID")],
    "书": [N("书", "READABLE")],
    "桌子": [N("桌子", "OBJECT")],
    "石头": [N("石头", "OBJECT")],
    "学校": [N("学校", "PLACE")],
    "家": [N("家", "PLACE")],
    "花": [N("花", "PLANT")],
    # ---- 动词 ----
    "吃": [V("吃", "吃", ["AGENT", "PATIENT"], ["PATIENT"])],
    "喝": [V("喝", "喝", ["AGENT", "PATIENT"], ["PATIENT"])],
    "跑": [V("跑", "跑", ["AGENT"], [])],
    # ★多义：同形、同框架，选择限制不同
    "看": [V("看", "看/read", ["AGENT", "PATIENT"], ["PATIENT"]),
           V("看", "看/visit", ["AGENT", "PATIENT"], ["PATIENT"])],
    "喜欢": [V("喜欢", "喜欢", ["EXPERIENCER", "STIMULUS"], ["STIMULUS"])],
    "去": [V("去", "去", ["AGENT", "GOAL"], [])],
    "给": [V("给", "给", ["AGENT", "RECIPIENT", "PATIENT"], [])],
    "有": [V("有", "有", ["POSSESSOR", "POSSESSED"], [])],
    "买": [V("买", "买", ["AGENT", "PATIENT"], ["PATIENT"])],
    # ---- 形容词 ----
    "红": [A()], "大": [A()], "小": [A()], "好": [A()], "忙": [A()], "高兴": [A()],
    # ---- 副词 / 助词 / 介词 ----
    "不": [ADV("NEG")], "很": [ADV("DEGREE")],
    "昨天": [ADV("TIME")], "今天": [ADV("TIME")],
    "了": [PART("PERF")], "的": [PART("MOD")],
    "在": [P()], "把": [P()], "被": [P()],
}

# 供只关心"词形→单一条目"的代码使用（名词属性、词性判断）
LEX = {f: es[0] for f, es in ENTRIES.items()}
VERB_FRAME = {e["sense"]: e["frame"]
              for es in ENTRIES.values() for e in es if e["pos"] == "V"}
VOPT = {e["sense"]: e["opt"]
        for es in ENTRIES.values() for e in es if e["pos"] == "V"}
SENSE_FORM = {e["sense"]: f
              for f, es in ENTRIES.items() for e in es if e["pos"] == "V"}
PROP_KEYS = ("animate", "edible", "liquid", "place")
