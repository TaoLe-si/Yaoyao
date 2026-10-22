# -*- coding: utf-8 -*-
"""训练/留出数据 v0.2。

v0.2 增加「看」的多义竞争：
  看/read  —— PATIENT 必须可读（书）
  看/visit —— PATIENT 必须是地点（学校）
同一句法框架，two candidates；exact match 必须选对义项。
None 表示该句句义不成立（REJECT）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from semantics import entity, event, state  # noqa: E402


def E(pred, roles=None, comps=None, **mods):
    return event(pred, roles or {}, comps=comps, **mods)


def S(pred, theme, **mods):
    return state(pred, {"THEME": theme}, **mods)


def SL(pred, theme, loc, **mods):
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
    ("我喜欢喝水", E("喜欢", {"EXPERIENCER": "我"},
                     comps={"STIMULUS": E("喝", {"AGENT": "我", "PATIENT": "水"})})),
    ("妈妈说的书", entity("书",
                       comps={"REL": E("说", {"AGENT": "妈妈", "PATIENT": "书"})},
                       REL_ROLE="PATIENT")),
    ("妈妈看他说的书", E("看/read", {
        "AGENT": "妈妈",
        "PATIENT": entity("书",
                          comps={"REL": E("说", {"AGENT": "他", "PATIENT": "书"})},
                          REL_ROLE="PATIENT")})),
    # ---- ATTR / QUANT / 主语关系化 ----
    ("我说话", E("说话", {"AGENT": "我"})),
    # ---- 时态/体 ----
    ("我吃过饭", E("吃", {"AGENT": "我", "PATIENT": "饭"}, EXP=True)),
    ("我看着书", E("看/read", {"AGENT": "我", "PATIENT": "书"}, PROG=True)),
    ("我正在吃饭", E("吃", {"AGENT": "我", "PATIENT": "饭"}, PROG=True)),
    ("我吃过饭了", E("吃", {"AGENT": "我", "PATIENT": "饭"}, EXP=True, PERF=True)),
    ("我喜欢过他", E("喜欢", {"EXPERIENCER": "我", "STIMULUS": "他"}, EXP=True)),
    ("老师看红的花", E("看/look", {
        "AGENT": "老师", "PATIENT": entity("花", ATTR="红")})),
    ("老师看说话的人", E("看/look", {
        "AGENT": "老师",
        "PATIENT": entity("人",
                          comps={"REL": E("说话", {"AGENT": "人"})},
                          REL_ROLE="AGENT")})),
    # ---- 看：两个义项 ----
    ("老师看书", E("看/read", {"AGENT": "老师", "PATIENT": "书"})),
    ("你看书", E("看/read", {"AGENT": "你", "PATIENT": "书"})),
    ("我今天看书", E("看/read", {"AGENT": "我", "PATIENT": "书"}, TIME=True)),
    ("老师看学校", E("看/visit", {"AGENT": "老师", "PATIENT": "学校"})),
    ("我看学校", E("看/visit", {"AGENT": "我", "PATIENT": "学校"})),
    ("老师看桌子", E("看/look", {"AGENT": "老师", "PATIENT": "桌子"})),
    ("我看石头", E("看/look", {"AGENT": "我", "PATIENT": "石头"})),
    # ---- 打：两个义项都可能成立，靠偏好消歧 ----
    ("我打球", E("打/play", {"AGENT": "我", "PATIENT": "球"})),
    ("他打球", E("打/play", {"AGENT": "他", "PATIENT": "球"})),
    ("我打猫", E("打/hit", {"AGENT": "我", "PATIENT": "猫"})),
    ("老师打猫", E("打/hit", {"AGENT": "老师", "PATIENT": "猫"})),
    ("老师打桌子", E("打/hit", {"AGENT": "老师", "PATIENT": "桌子"})),
    ("我打石头", E("打/hit", {"AGENT": "我", "PATIENT": "石头"})),
    # ---- 结构覆盖：并列 / 疑问 / 否定形容词 / 数量短语 ----
    ("我和小明吃饭", E("吃", {"AGENT": "我+小明", "PATIENT": "饭"})),
    ("你吃饭吗", E("吃", {"AGENT": "你", "PATIENT": "饭"}, Q=True)),
    ("老师看书吗", E("看/read", {"AGENT": "老师", "PATIENT": "书"}, Q=True)),
    ("花不红", S("红", "花", NEG=True)),
    ("老师不高兴", S("高兴", "老师", NEG=True)),
    ("我吃一个苹果", E("吃", {"AGENT": "我",
                             "PATIENT": entity("一个苹果", QUANT="一个")})),
    ("妈妈买一本书", E("买", {"AGENT": "妈妈",
                             "PATIENT": entity("一本书", QUANT="一本")})),
    ("一只猫吃鱼", E("吃", {"AGENT": entity("一只猫", QUANT="一只"),
                           "PATIENT": "鱼"})),
    # ---- 其他动词 ----
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
    ("小明跑", E("跑", {"AGENT": "小明"})),
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
    ("花看鸟", None),
    ("桌子很高兴", None),
    ("老师给桌子书", None),
    ("桌子很忙", None),
    ("桌子打球", None),
    ("苹果打猫", None),
    # ---- 结构覆盖负例 ----
    ("我和桌子吃饭", None),
    ("石头吃饭吗", None),
    ("石头不高兴", None),
    ("我吃一个石头", None),
    ("一个石头吃鱼", None),
    # ---- 在字句：LOC 必须是地点 ----
    ("猫在桌子", None),
    ("书在水", None),
    ("老师在苹果", None),
    ("桌子很聪明", None),
    ("我喜欢吃石头", None),
    ("我跑的猫", None),
    # ---- 体标记：重复 / 冲突 / 状态动词 / 无谓词 ----
    ("我吃了了饭", None),
    ("我吃着过饭", None),
    ("我吃了过饭", None),
    ("我吃着饭了", None),
    ("花红着", None),
    ("我正在", None),
    ("我饭了", None),
    ("我正在喜欢他", None),
    ("我喜欢着他", None),
]

HELDOUT_VALID = [
    # ---- 看：义项消歧 ----
    ("妈妈看书", E("看/read", {"AGENT": "妈妈", "PATIENT": "书"})),
    ("他看书", E("看/read", {"AGENT": "他", "PATIENT": "书"})),
    ("小明看学校", E("看/visit", {"AGENT": "小明", "PATIENT": "学校"})),
    ("狗看学校", E("看/visit", {"AGENT": "狗", "PATIENT": "学校"})),
    ("我看桌子", E("看/look", {"AGENT": "我", "PATIENT": "桌子"})),
    # ---- 其他 ----
    ("狗吃苹果", E("吃", {"AGENT": "狗", "PATIENT": "苹果"})),
    ("妈妈喝水", E("喝", {"AGENT": "妈妈", "PATIENT": "水"})),
    ("老师喜欢花", E("喜欢", {"EXPERIENCER": "老师", "STIMULUS": "花"})),
    ("小明去学校", E("去", {"AGENT": "小明", "GOAL": "学校"})),
    ("猫看书", E("看/read", {"AGENT": "猫", "PATIENT": "书"})),
    ("我买苹果", E("买", {"AGENT": "我", "PATIENT": "苹果"})),
    ("妈妈有书", E("有", {"POSSESSOR": "妈妈", "POSSESSED": "书"})),
    ("他给妈妈水", E("给", {"AGENT": "他", "RECIPIENT": "妈妈", "PATIENT": "水"})),
    ("狗在家", SL("在", "狗", "家")),
    ("花很小", S("小", "花", DEGREE=True)),
    ("大的花", entity("花", ATTR="大")),
    ("鸡吃苹果", E("吃", {"AGENT": "鸡", "PATIENT": "苹果"})),
    ("鱼喝水", E("喝", {"AGENT": "鱼", "PATIENT": "水"})),
    ("猫吃苹果", E("吃", {"AGENT": "猫", "PATIENT": "苹果"})),
    ("我不看书", E("看/read", {"AGENT": "我", "PATIENT": "书"}, NEG=True)),
    ("妈妈把水喝了", E("喝", {"AGENT": "妈妈", "PATIENT": "水"}, PERF=True, BA=True)),
    ("他很高兴", S("高兴", "他", DEGREE=True)),
    ("鸟吃苹果", E("吃", {"AGENT": "鸟", "PATIENT": "苹果"})),
    ("老师喝水", E("喝", {"AGENT": "老师", "PATIENT": "水"})),
    ("小明买书", E("买", {"AGENT": "小明", "PATIENT": "书"})),
    ("我给他苹果", E("给", {"AGENT": "我", "RECIPIENT": "他", "PATIENT": "苹果"})),
    ("妈妈给老师书", E("给", {"AGENT": "妈妈", "RECIPIENT": "老师", "PATIENT": "书"})),
    ("狗去学校", E("去", {"AGENT": "狗", "GOAL": "学校"})),
    ("老师在家", SL("在", "老师", "家")),
    ("花很大", S("大", "花", DEGREE=True)),
    ("大的书", entity("书", ATTR="大")),
    ("小的苹果", entity("苹果", ATTR="小")),
    # ---- 把 / 被 的结构覆盖 ----
    ("水被我喝了", E("喝", {"AGENT": "我", "PATIENT": "水"}, PERF=True, BEI=True)),
    ("苹果被妈妈买了", E("买", {"AGENT": "妈妈", "PATIENT": "苹果"}, PERF=True, BEI=True)),
    ("我把水喝了", E("喝", {"AGENT": "我", "PATIENT": "水"}, PERF=True, BA=True)),
    ("妈妈把苹果买了", E("买", {"AGENT": "妈妈", "PATIENT": "苹果"}, PERF=True, BA=True)),
    # ---- 打：留出义项偏好 ----
    ("小明打球", E("打/play", {"AGENT": "小明", "PATIENT": "球"})),
    ("老师打球", E("打/play", {"AGENT": "老师", "PATIENT": "球"})),
    ("狗打球", E("打/play", {"AGENT": "狗", "PATIENT": "球"})),
    ("妈妈打猫", E("打/hit", {"AGENT": "妈妈", "PATIENT": "猫"})),
    # ---- 结构覆盖 held-out ----
    ("我和老师看书", E("看/read", {"AGENT": "我+老师", "PATIENT": "书"})),
    ("妈妈喝水吗", E("喝", {"AGENT": "妈妈", "PATIENT": "水"}, Q=True)),
    ("苹果不红", S("红", "苹果", NEG=True)),
    ("妈妈不高兴", S("高兴", "妈妈", NEG=True)),
    ("我买一个苹果", E("买", {"AGENT": "我",
                             "PATIENT": entity("一个苹果", QUANT="一个")})),
    ("一只狗吃苹果", E("吃", {"AGENT": entity("一只狗", QUANT="一只"),
                             "PATIENT": "苹果"})),
    ("老师和妈妈看学校", E("看/visit", {"AGENT": "老师+妈妈", "PATIENT": "学校"})),
]

HELDOUT_INVALID = [
    # ---- 看：两个义项都不成立 ----
    ("苹果看学校", None),
    ("桌子看书", None),
    # ---- 其他 ----
    ("苹果吃鱼", None),
    ("石头喝水", None),
    ("书吃饭", None),
    ("桌子去学校", None),
    ("我喝石头", None),
    ("老师吃水", None),
    ("猫喝石头", None),
    ("小明去书", None),
    ("苹果给小明书", None),
    ("石头喜欢猫", None),
    ("我吃书", None),
    ("狗吃桌子", None),
    ("石头买书", None),
    ("妈妈喝饭", None),
    ("猫吃水", None),
    ("石头给老师书", None),
    ("老师给石头书", None),
    ("桌子有书", None),
    ("花去学校", None),
    ("苹果喜欢书", None),
    ("石头很高兴", None),
    ("石头很忙", None),
    # ---- 把 / 被 的负例 ----
    ("石头被我吃了", None),
    ("我被石头吃了", None),
    ("我把石头吃了", None),
    ("石头把苹果吃了", None),
    ("苹果打球", None),
    ("桌子打猫", None),
    # ---- 结构覆盖负例 ----
    ("我和石头看书", None),
    ("桌子喝水吗", None),
    ("苹果不高兴", None),
    ("一个石头喝水", None),
    ("我和苹果吃饭", None),
]

TRAIN = TRAIN_VALID + TRAIN_INVALID
HELDOUT = HELDOUT_VALID + HELDOUT_INVALID
