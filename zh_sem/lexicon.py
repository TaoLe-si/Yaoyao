# -*- coding: utf-8 -*-
"""手写字词语义种子（45 条）。

表示法：
  form  : 词形
  pos   : N / PRON / V / A / ADV / P / PART
  cls   : 粗粒度语义类（只作一个特征，不承载全部语义）
  props : 正交属性 animate / edible / liquid / place（逐词手写）
  frame : 仅动词；有序论元框架，第一项是主语角色
  opt   : 仅动词；可省略的论元

★v0.1 修订（专家评审 A1）：属性从"语义类簇"里拆出来。
  ANIMAL_EDIBLE 这种类会把"是动物"和"可吃"绑死，导致 role_edible(AGENT)
  被负例压负、误伤鸡/鱼。现在每个词独立给四个布尔位。
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

LEX = {
    # ---- 代词 ----
    "我":   dict(pos="PRON", cls="HUMAN", props=NOUN_PROPS["我"]),
    "你":   dict(pos="PRON", cls="HUMAN", props=NOUN_PROPS["你"]),
    "他":   dict(pos="PRON", cls="HUMAN", props=NOUN_PROPS["他"]),
    # ---- 名词 ----
    "小明": dict(pos="N", cls="HUMAN", props=NOUN_PROPS["小明"]),
    "妈妈": dict(pos="N", cls="HUMAN", props=NOUN_PROPS["妈妈"]),
    "老师": dict(pos="N", cls="HUMAN", props=NOUN_PROPS["老师"]),
    "猫":   dict(pos="N", cls="ANIMAL", props=NOUN_PROPS["猫"]),
    "狗":   dict(pos="N", cls="ANIMAL", props=NOUN_PROPS["狗"]),
    "鸟":   dict(pos="N", cls="ANIMAL", props=NOUN_PROPS["鸟"]),
    "鸡":   dict(pos="N", cls="ANIMAL", props=NOUN_PROPS["鸡"]),
    "鱼":   dict(pos="N", cls="ANIMAL", props=NOUN_PROPS["鱼"]),
    "苹果": dict(pos="N", cls="FRUIT", props=NOUN_PROPS["苹果"]),
    "饭":   dict(pos="N", cls="FOOD", props=NOUN_PROPS["饭"]),
    "水":   dict(pos="N", cls="LIQUID", props=NOUN_PROPS["水"]),
    "书":   dict(pos="N", cls="READABLE", props=NOUN_PROPS["书"]),
    "桌子": dict(pos="N", cls="OBJECT", props=NOUN_PROPS["桌子"]),
    "石头": dict(pos="N", cls="OBJECT", props=NOUN_PROPS["石头"]),
    "学校": dict(pos="N", cls="PLACE", props=NOUN_PROPS["学校"]),
    "家":   dict(pos="N", cls="PLACE", props=NOUN_PROPS["家"]),
    "花":   dict(pos="N", cls="PLANT", props=NOUN_PROPS["花"]),
    # ---- 动词（frame 有序，第一项为主语角色）----
    "吃":   dict(pos="V", frame=["AGENT", "PATIENT"], opt=["PATIENT"]),
    "喝":   dict(pos="V", frame=["AGENT", "PATIENT"], opt=["PATIENT"]),
    "跑":   dict(pos="V", frame=["AGENT"], opt=[]),
    "看":   dict(pos="V", frame=["AGENT", "PATIENT"], opt=["PATIENT"]),
    "喜欢": dict(pos="V", frame=["EXPERIENCER", "STIMULUS"], opt=["STIMULUS"]),
    "去":   dict(pos="V", frame=["AGENT", "GOAL"], opt=[]),
    "给":   dict(pos="V", frame=["AGENT", "RECIPIENT", "PATIENT"], opt=[]),
    "有":   dict(pos="V", frame=["POSSESSOR", "POSSESSED"], opt=[]),
    "买":   dict(pos="V", frame=["AGENT", "PATIENT"], opt=["PATIENT"]),
    # ---- 形容词 ----
    "红":   dict(pos="A"), "大": dict(pos="A"), "小": dict(pos="A"),
    "好":   dict(pos="A"), "忙": dict(pos="A"), "高兴": dict(pos="A"),
    # ---- 副词 ----
    "不":   dict(pos="ADV", kind="NEG"),
    "很":   dict(pos="ADV", kind="DEGREE"),
    "昨天": dict(pos="ADV", kind="TIME"),
    "今天": dict(pos="ADV", kind="TIME"),
    # ---- 助词 ----
    "了":   dict(pos="PART", kind="PERF"),
    "的":   dict(pos="PART", kind="MOD"),
    # ---- 介词 ----
    "在":   dict(pos="P"),
    "把":   dict(pos="P"),
    "被":   dict(pos="P"),
}

VERB_FRAME = {f: i["frame"] for f, i in LEX.items() if i["pos"] == "V"}
VOPT = {f: i["opt"] for f, i in LEX.items() if i["pos"] == "V"}
PROP_KEYS = ("animate", "edible", "liquid", "place")
