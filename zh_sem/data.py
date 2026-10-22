# -*- coding: utf-8 -*-
"""训练/留出数据。None 表示该句句义不成立（REJECT）。

正例给出完整句义帧；负例是语义选择限制被违反、或句法不完整。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from semantics import entity, event, state  # noqa: E402


def E(pred, roles=None, **mods):
    return event(pred, roles or {}, **mods)


def S(pred, theme, **mods):
    return state(pred, {"THEME": theme}, **mods)


def SL(pred, theme, loc, **mods):
    """在字句：LOC 是谓词的一个论元，放 roles 而不是 mods。"""
    return state(pred, {"THEME": theme, "LOC": loc}, **mods)


TRAIN_VALID = [
    ("我吃饭", E("吃", {"AGENT": "我", "PATIENT": "饭"})),
    ("你喝水", E("喝", {"AGENT": "你", "PATIENT": "水"})),
    ("猫跑", E("跑", {"AGENT": "猫"})),
    ("狗跑", E("跑", {"AGENT": "狗"})),
    ("鸡吃", E("吃", {"AGENT": "鸡"})),
    ("鱼喝", E("喝", {"AGENT": "鱼"})),
    ("我吃鸡", E("吃", {"AGENT": "我", "PATIENT": "鸡"})),
    ("猫吃鱼", E("吃", {"AGENT": "猫", "PATIENT": "鱼"})),
    ("鸡喝水", E("喝", {"AGENT": "鸡", "PATIENT": "水"})),
    ("猫喝水", E("喝", {"AGENT": "猫", "PATIENT": "水"})),
    ("我吃苹果", E("吃", {"AGENT": "我", "PATIENT": "苹果"})),
    ("小明喜欢苹果", E("喜欢", {"EXPERIENCER": "小明", "STIMULUS": "苹果"})),
    ("猫喜欢鱼", E("喜欢", {"EXPERIENCER": "猫", "STIMULUS": "鱼"})),
    ("老师看书", E("看", {"AGENT": "老师", "PATIENT": "书"})),
    ("我买书", E("买", {"AGENT": "我", "PATIENT": "书"})),
    ("他买苹果", E("买", {"AGENT": "他", "PATIENT": "苹果"})),
    ("妈妈买苹果", E("买", {"AGENT": "妈妈", "PATIENT": "苹果"})),
    ("老师有书", E("有", {"POSSESSOR": "老师", "POSSESSED": "书"})),
    ("我给他书", E("给", {"AGENT": "我", "RECIPIENT": "他", "PATIENT": "书"})),
    ("妈妈给小明苹果", E("给", {"AGENT": "妈妈", "RECIPIENT": "小明", "PATIENT": "苹果"})),
    ("我去学校", E("去", {"AGENT": "我", "GOAL": "学校"})),
    ("猫在家", SL("在", "猫", "家")),
    ("花很红", S("红", "花", DEGREE=True)),
    ("苹果很大", S("大", "苹果", DEGREE=True)),
    ("红的花", entity("花", ATTR="红")),
    ("大的苹果", entity("苹果", ATTR="大")),
    ("我吃了饭", E("吃", {"AGENT": "我", "PATIENT": "饭"}, PERF=True)),
    ("我不喝水", E("喝", {"AGENT": "我", "PATIENT": "水"}, NEG=True)),
    ("我在家吃饭", E("吃", {"AGENT": "我", "PATIENT": "饭"}, LOC="家")),
    ("我把饭吃了", E("吃", {"AGENT": "我", "PATIENT": "饭"}, PERF=True, BA=True)),
    ("饭被我吃了", E("吃", {"AGENT": "我", "PATIENT": "饭"}, PERF=True, BEI=True)),
    ("妈妈很高兴", S("高兴", "妈妈", DEGREE=True)),
    ("老师很忙", S("忙", "老师", DEGREE=True)),
    ("狗不跑", E("跑", {"AGENT": "狗"}, NEG=True)),
    ("我今天看书", E("看", {"AGENT": "我", "PATIENT": "书"}, TIME=True)),
    ("小明跑", E("跑", {"AGENT": "小明"})),
    ("你看书", E("看", {"AGENT": "你", "PATIENT": "书"})),
]

TRAIN_INVALID = [
    ("石头吃饭", None),
    ("苹果喝水", None),
    ("桌子跑", None),
    ("饭看书", None),
    ("水吃饭", None),
    ("书跑", None),
    ("石头喜欢鱼", None),
    ("苹果去学校", None),
    ("桌子给小明书", None),
    ("花有书", None),
    ("石头买苹果", None),
    ("我吃石头", None),
    ("猫喝饭", None),
    ("老师吃桌子", None),
    ("鱼吃石头", None),
    ("小明喝苹果", None),
    ("我去桌子", None),
    ("猫给狗", None),
    ("苹果很跑", None),
    ("我喝书", None),
    ("书喜欢苹果", None),
    ("石头不跑", None),
]

HELDOUT_VALID = [
    ("狗吃苹果", E("吃", {"AGENT": "狗", "PATIENT": "苹果"})),
    ("妈妈喝水", E("喝", {"AGENT": "妈妈", "PATIENT": "水"})),
    ("老师喜欢花", E("喜欢", {"EXPERIENCER": "老师", "STIMULUS": "花"})),
    ("小明去学校", E("去", {"AGENT": "小明", "GOAL": "学校"})),
    ("猫看书", E("看", {"AGENT": "猫", "PATIENT": "书"})),
    ("我买苹果", E("买", {"AGENT": "我", "PATIENT": "苹果"})),
    ("妈妈有书", E("有", {"POSSESSOR": "妈妈", "POSSESSED": "书"})),
    ("他给妈妈水", E("给", {"AGENT": "他", "RECIPIENT": "妈妈", "PATIENT": "水"})),
    ("狗在家", SL("在", "狗", "家")),
    ("花很小", S("小", "花", DEGREE=True)),
    ("大的花", entity("花", ATTR="大")),
    ("鸡吃苹果", E("吃", {"AGENT": "鸡", "PATIENT": "苹果"})),
    ("鱼喝水", E("喝", {"AGENT": "鱼", "PATIENT": "水"})),
    ("猫吃苹果", E("吃", {"AGENT": "猫", "PATIENT": "苹果"})),
    ("我不看书", E("看", {"AGENT": "我", "PATIENT": "书"}, NEG=True)),
    ("妈妈把水喝了", E("喝", {"AGENT": "妈妈", "PATIENT": "水"}, PERF=True, BA=True)),
    ("他很高兴", S("高兴", "他", DEGREE=True)),
]

HELDOUT_INVALID = [
    ("苹果吃鱼", None),
    ("石头喝水", None),
    ("书吃饭", None),
    ("桌子去学校", None),
    ("花看鸟", None),
    ("我喝石头", None),
    ("老师吃水", None),
    ("猫喝石头", None),
    ("小明去书", None),
    ("苹果给小明书", None),
    ("石头喜欢猫", None),
    ("我吃书", None),
    ("狗吃桌子", None),
    ("石头买书", None),
]

TRAIN = TRAIN_VALID + TRAIN_INVALID
HELDOUT = HELDOUT_VALID + HELDOUT_INVALID
