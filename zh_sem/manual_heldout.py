# -*- coding: utf-8 -*-
"""独立人工 held-out（不调用验证器、不生成伪标签）。

★这份 gold 是手写的语言学判断，不是 ROLE_REQ 的输出。
句子全部使用 v0.3 新增词表（未出现在 TRAIN/HELDOUT 中），因此同时是
"新词泛化"测试。评测时另跑验证器，比较"模型 vs 人"与"验证器 vs 人"。
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


MANUAL_VALID = [
    # ---- 新名词作论元 ----
    ("狗喝水", E("喝", {"AGENT": "狗", "PATIENT": "水"})),
    ("妈妈吃苹果", E("吃", {"AGENT": "妈妈", "PATIENT": "苹果"})),
    ("老师吃苹果", E("吃", {"AGENT": "老师", "PATIENT": "苹果"})),
    ("小明喝水", E("喝", {"AGENT": "小明", "PATIENT": "水"})),
    ("姐姐看书", E("看/read", {"AGENT": "姐姐", "PATIENT": "书"})),
    ("哥哥看学校", E("看/visit", {"AGENT": "哥哥", "PATIENT": "学校"})),
    ("老师看石头", E("看/look", {"AGENT": "老师", "PATIENT": "石头"})),
    ("妈妈看衣服", E("看/look", {"AGENT": "妈妈", "PATIENT": "衣服"})),
    ("老师看水", E("看/look", {"AGENT": "老师", "PATIENT": "水"})),
    ("弟弟看桌子", E("看/look", {"AGENT": "弟弟", "PATIENT": "桌子"})),
    ("医生打球", E("打/play", {"AGENT": "医生", "PATIENT": "球"})),
    # ---- 新动词 ----
    ("学生写书", E("写", {"AGENT": "学生", "PATIENT": "书"})),
    ("工人读报纸", E("读", {"AGENT": "工人", "PATIENT": "报纸"})),
    ("朋友来学校", E("来", {"AGENT": "朋友", "GOAL": "学校"})),
    ("弟弟回医院", E("回", {"AGENT": "弟弟", "GOAL": "医院"})),
    ("妹妹问老师", E("问", {"AGENT": "妹妹", "RECIPIENT": "老师"})),
    ("爷爷教学生", E("教", {"AGENT": "爷爷", "RECIPIENT": "学生"})),
    ("奶奶送朋友书", E("送", {"AGENT": "奶奶", "RECIPIENT": "朋友", "PATIENT": "书"})),
    ("牛吃草", E("吃", {"AGENT": "牛", "PATIENT": "草"})),
    ("羊喝水", E("喝", {"AGENT": "羊", "PATIENT": "水"})),
    ("马吃苹果", E("吃", {"AGENT": "马", "PATIENT": "苹果"})),
    ("兔吃菜", E("吃", {"AGENT": "兔", "PATIENT": "菜"})),
    ("猴吃香蕉", E("吃", {"AGENT": "猴", "PATIENT": "香蕉"})),
    ("医生喝咖啡", E("喝", {"AGENT": "医生", "PATIENT": "咖啡"})),
    ("学生喝牛奶", E("喝", {"AGENT": "学生", "PATIENT": "牛奶"})),
    ("妈妈买香蕉", E("买", {"AGENT": "妈妈", "PATIENT": "香蕉"})),
    ("老师卖书", E("卖", {"AGENT": "老师", "PATIENT": "书"})),
    ("工人找笔", E("找", {"AGENT": "工人", "PATIENT": "笔"})),
    ("朋友洗衣服", E("洗", {"AGENT": "朋友", "PATIENT": "衣服"})),
    ("妹妹开窗", E("开", {"AGENT": "妹妹", "PATIENT": "窗"})),
    ("弟弟关电视", E("关", {"AGENT": "弟弟", "PATIENT": "电视"})),
    ("爷爷等朋友", E("等", {"AGENT": "爷爷", "PATIENT": "朋友"})),
    ("医生认识老师", E("认识", {"EXPERIENCER": "医生", "STIMULUS": "老师"})),
    ("妈妈爱爸爸", E("爱", {"EXPERIENCER": "妈妈", "STIMULUS": "爸爸"})),
    ("我喜欢吃苹果", E("喜欢", {"EXPERIENCER": "我"},
                      comps={"STIMULUS": E("吃", {"AGENT": "我", "PATIENT": "苹果"})})),
    # ---- 新动词 SV ----
    ("姐姐唱", E("唱", {"AGENT": "姐姐"})),
    ("哥哥跳", E("跳", {"AGENT": "哥哥"})),
    ("弟弟走", E("走", {"AGENT": "弟弟"})),
    ("妹妹睡", E("睡", {"AGENT": "妹妹"})),
    ("爷爷站", E("站", {"AGENT": "爷爷"})),
    # ---- 新形容词 / 在字句 ----
    ("教室很干净", S("干净", "教室", DEGREE=True)),
    ("公园很安静", S("安静", "公园", DEGREE=True)),
    ("衣服很漂亮", S("漂亮", "衣服", DEGREE=True)),
    ("电脑很新", S("新", "电脑", DEGREE=True)),
    ("茶很热", S("热", "茶", DEGREE=True)),
    ("医院在车站", SL("在", "医院", "车站")),
    # ---- 把 / 被 ----
    ("老师把书卖了", E("卖", {"AGENT": "老师", "PATIENT": "书"}, PERF=True, BA=True)),
    ("苹果被学生吃了", E("吃", {"AGENT": "学生", "PATIENT": "苹果"}, PERF=True, BEI=True)),
    # ---- 否定 / 疑问 / 并列 / 数量 ----
    ("狗不睡", E("睡", {"AGENT": "狗"}, NEG=True)),
    ("花不漂亮", S("漂亮", "花", NEG=True)),
    ("你打球吗", E("打/play", {"AGENT": "你", "PATIENT": "球"}, Q=True)),
    ("妈妈看书吗", E("看/read", {"AGENT": "妈妈", "PATIENT": "书"}, Q=True)),
    ("我和姐姐打球", E("打/play", {"AGENT": "我+姐姐", "PATIENT": "球"})),
    ("一只猫喝水", E("喝", {"AGENT": "一只猫", "PATIENT": "水"})),
    ("弟弟吃一个苹果", E("吃", {"AGENT": "弟弟", "PATIENT": "一个苹果"})),
    ("我昨天去了学校", E("去", {"AGENT": "我", "GOAL": "学校"}, TIME=True, PERF=True)),
]

MANUAL_INVALID = [
    ("狗吃石头", None),
    ("妈妈喝面条", None),
    ("电视打球", None),
    ("石头写书", None),
    ("苹果读报纸", None),
    ("桌子来学校", None),
    ("花问老师", None),
    ("水教学生", None),
    ("书送朋友", None),
    ("电视买苹果", None),
    ("我吃一个电视", None),
    ("一个电视喝水", None),
    ("狗在桌子", None),
    ("苹果很聪明", None),
    ("窗很高兴", None),
    ("我和石头吃饭", None),
    ("我喝面包", None),
    ("老师吃咖啡", None),
    ("学生去书", None),
    ("爷爷回苹果", None),
    ("苹果被石头吃了", None),
    ("电视把苹果吃了", None),
    ("老师把石头吃了", None),
    ("花喝水", None),
]

MANUAL = MANUAL_VALID + MANUAL_INVALID

# 覆盖缺口：这些句子在真实汉语里成立，但当前词典/语法表示不了。
# 不计入准确率；单独报告，用来证明验证器 != 真理。
MANUAL_COVERAGE_GAPS = [
    ("他说的书", "关系从句 构式缺失"),
]
