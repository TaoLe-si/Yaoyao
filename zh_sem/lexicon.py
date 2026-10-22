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

def _p(animate=False, edible=False, liquid=False, place=False,
       readable=False, sport=False):
    return dict(animate=animate, edible=edible, liquid=liquid, place=place,
                readable=readable, sport=sport)


NOUN_PROPS = {
    "我":    _p(animate=True),
    "你":    _p(animate=True),
    "他":    _p(animate=True),
    "小明":  _p(animate=True),
    "妈妈":  _p(animate=True),
    "老师":  _p(animate=True),
    "猫":    _p(animate=True),
    "狗":    _p(animate=True),
    "鸟":    _p(animate=True),
    "鸡":    _p(animate=True, edible=True),
    "鱼":    _p(animate=True, edible=True),
    "苹果":  _p(edible=True),
    "饭":    _p(edible=True),
    "水":    _p(liquid=True),
    "书":    _p(readable=True),
    "桌子":  _p(),
    "石头":  _p(),
    "学校":  _p(place=True),
    "家":    _p(place=True),
    "花":    _p(),
    "球":    _p(sport=True),
    # ---- 扩展：人物 ----
    "爸爸": _p(animate=True), "哥哥": _p(animate=True), "姐姐": _p(animate=True),
    "弟弟": _p(animate=True), "妹妹": _p(animate=True), "爷爷": _p(animate=True),
    "奶奶": _p(animate=True), "医生": _p(animate=True), "学生": _p(animate=True),
    "工人": _p(animate=True), "朋友": _p(animate=True), "同学": _p(animate=True),
    # ---- 扩展：动物 ----
    "牛": _p(animate=True, edible=True), "羊": _p(animate=True, edible=True),
    "马": _p(animate=True, edible=True), "猪": _p(animate=True, edible=True),
    "兔": _p(animate=True, edible=True), "鸭": _p(animate=True, edible=True),
    "鹅": _p(animate=True, edible=True), "虎": _p(animate=True),
    "象": _p(animate=True), "熊": _p(animate=True), "猴": _p(animate=True),
    # ---- 扩展：水果 / 食物 / 液体 ----
    "香蕉": _p(edible=True), "橘子": _p(edible=True), "西瓜": _p(edible=True),
    "葡萄": _p(edible=True), "梨": _p(edible=True), "桃": _p(edible=True),
    "面包": _p(edible=True), "面条": _p(edible=True), "米饭": _p(edible=True),
    "肉": _p(edible=True), "鸡蛋": _p(edible=True), "菜": _p(edible=True),
    "牛奶": _p(liquid=True), "茶": _p(liquid=True),
    "咖啡": _p(liquid=True), "果汁": _p(liquid=True),
    # ---- 扩展：物体 / 可读物 ----
    "笔": _p(), "书包": _p(), "衣服": _p(), "鞋": _p(), "帽子": _p(),
    "手机": _p(), "电脑": _p(), "电视": _p(), "门": _p(), "窗": _p(),
    "报纸": _p(readable=True), "杂志": _p(readable=True),
    # ---- 扩展：地点 / 植物 ----
    "公园": _p(place=True), "医院": _p(place=True), "商店": _p(place=True),
    "银行": _p(place=True), "车站": _p(place=True), "教室": _p(place=True),
    "图书馆": _p(place=True), "树": _p(), "草": _p(),
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


def CONJ():
    return dict(pos="CONJ")


def NUM():
    return dict(pos="NUM")


def CL():
    return dict(pos="CL")


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
    "球": [N("球", "SPORT")],
    # ---- 动词 ----
    "吃": [V("吃", "吃", ["AGENT", "PATIENT"], ["PATIENT"])],
    "喝": [V("喝", "喝", ["AGENT", "PATIENT"], ["PATIENT"])],
    "跑": [V("跑", "跑", ["AGENT"], [])],
    # ★多义：同形、同框架，选择限制不同
    "看": [V("看", "看/read", ["AGENT", "PATIENT"], ["PATIENT"]),
           V("看", "看/visit", ["AGENT", "PATIENT"], ["PATIENT"])],
    # ★第二种多义：两个义项都可能成立，靠词汇偏好消歧
    "打": [V("打", "打/hit", ["AGENT", "PATIENT"], ["PATIENT"]),
           V("打", "打/play", ["AGENT", "PATIENT"], ["PATIENT"])],
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
    "了": [PART("PERF")], "的": [PART("MOD")], "吗": [PART("Q")],
    "在": [P()], "把": [P()], "被": [P()],
    "和": [CONJ()],
    "一": [NUM()],
    "个": [CL()], "本": [CL()], "只": [CL()],
    # ================= 扩展词表（60 名词 / 26 动词 / 16 形容词）=================
    "爸爸": [N("爸爸", "HUMAN")], "哥哥": [N("哥哥", "HUMAN")],
    "姐姐": [N("姐姐", "HUMAN")], "弟弟": [N("弟弟", "HUMAN")],
    "妹妹": [N("妹妹", "HUMAN")], "爷爷": [N("爷爷", "HUMAN")],
    "奶奶": [N("奶奶", "HUMAN")], "医生": [N("医生", "HUMAN")],
    "学生": [N("学生", "HUMAN")], "工人": [N("工人", "HUMAN")],
    "朋友": [N("朋友", "HUMAN")], "同学": [N("同学", "HUMAN")],
    "牛": [N("牛", "ANIMAL")], "羊": [N("羊", "ANIMAL")],
    "马": [N("马", "ANIMAL")], "猪": [N("猪", "ANIMAL")],
    "兔": [N("兔", "ANIMAL")], "鸭": [N("鸭", "ANIMAL")],
    "鹅": [N("鹅", "ANIMAL")], "虎": [N("虎", "ANIMAL")],
    "象": [N("象", "ANIMAL")], "熊": [N("熊", "ANIMAL")],
    "猴": [N("猴", "ANIMAL")],
    "香蕉": [N("香蕉", "FRUIT")], "橘子": [N("橘子", "FRUIT")],
    "西瓜": [N("西瓜", "FRUIT")], "葡萄": [N("葡萄", "FRUIT")],
    "梨": [N("梨", "FRUIT")], "桃": [N("桃", "FRUIT")],
    "面包": [N("面包", "FOOD")], "面条": [N("面条", "FOOD")],
    "米饭": [N("米饭", "FOOD")], "肉": [N("肉", "FOOD")],
    "鸡蛋": [N("鸡蛋", "FOOD")], "菜": [N("菜", "FOOD")],
    "牛奶": [N("牛奶", "LIQUID")], "茶": [N("茶", "LIQUID")],
    "咖啡": [N("咖啡", "LIQUID")], "果汁": [N("果汁", "LIQUID")],
    "笔": [N("笔", "OBJECT")], "书包": [N("书包", "OBJECT")],
    "衣服": [N("衣服", "OBJECT")], "鞋": [N("鞋", "OBJECT")],
    "帽子": [N("帽子", "OBJECT")], "手机": [N("手机", "OBJECT")],
    "电脑": [N("电脑", "OBJECT")], "电视": [N("电视", "OBJECT")],
    "门": [N("门", "OBJECT")], "窗": [N("窗", "OBJECT")],
    "报纸": [N("报纸", "READABLE")], "杂志": [N("杂志", "READABLE")],
    "公园": [N("公园", "PLACE")], "医院": [N("医院", "PLACE")],
    "商店": [N("商店", "PLACE")], "银行": [N("银行", "PLACE")],
    "车站": [N("车站", "PLACE")], "教室": [N("教室", "PLACE")],
    "图书馆": [N("图书馆", "PLACE")], "树": [N("树", "PLANT")],
    "草": [N("草", "PLANT")],
    # ---- 扩展动词 ----
    "写": [V("写", "写", ["AGENT", "PATIENT"], ["PATIENT"])],
    "读": [V("读", "读", ["AGENT", "PATIENT"], ["PATIENT"])],
    "说": [V("说", "说", ["AGENT", "PATIENT"], ["PATIENT"])],
    "听": [V("听", "听", ["AGENT", "PATIENT"], ["PATIENT"])],
    "唱": [V("唱", "唱", ["AGENT"], [])],
    "跳": [V("跳", "跳", ["AGENT"], [])],
    "走": [V("走", "走", ["AGENT"], [])],
    "来": [V("来", "来", ["AGENT", "GOAL"], [])],
    "回": [V("回", "回", ["AGENT", "GOAL"], [])],
    "睡": [V("睡", "睡", ["AGENT"], [])],
    "站": [V("站", "站", ["AGENT"], [])],
    "洗": [V("洗", "洗", ["AGENT", "PATIENT"], ["PATIENT"])],
    "开": [V("开", "开", ["AGENT", "PATIENT"], ["PATIENT"])],
    "关": [V("关", "关", ["AGENT", "PATIENT"], ["PATIENT"])],
    "找": [V("找", "找", ["AGENT", "PATIENT"], ["PATIENT"])],
    "等": [V("等", "等", ["AGENT", "PATIENT"], ["PATIENT"])],
    "问": [V("问", "问", ["AGENT", "RECIPIENT", "PATIENT"], [])],
    "教": [V("教", "教", ["AGENT", "RECIPIENT", "PATIENT"], [])],
    "学": [V("学", "学", ["AGENT", "PATIENT"], ["PATIENT"])],
    "做": [V("做", "做", ["AGENT", "PATIENT"], ["PATIENT"])],
    "用": [V("用", "用", ["AGENT", "PATIENT"], ["PATIENT"])],
    "送": [V("送", "送", ["AGENT", "RECIPIENT", "PATIENT"], [])],
    "卖": [V("卖", "卖", ["AGENT", "PATIENT"], ["PATIENT"])],
    "认识": [V("认识", "认识", ["EXPERIENCER", "STIMULUS"], ["STIMULUS"])],
    "爱": [V("爱", "爱", ["EXPERIENCER", "STIMULUS"], ["STIMULUS"])],
    "知道": [V("知道", "知道", ["EXPERIENCER", "STIMULUS"], ["STIMULUS"])],
    # ---- 扩展形容词 ----
    "新": [A()], "旧": [A()], "快": [A()], "慢": [A()], "高": [A()], "矮": [A()],
    "长": [A()], "短": [A()], "冷": [A()], "热": [A()], "干净": [A()], "脏": [A()],
    "漂亮": [A()], "聪明": [A()], "安静": [A()], "吵": [A()],
}

# 供只关心"词形→单一条目"的代码使用（名词属性、词性判断）
LEX = {f: es[0] for f, es in ENTRIES.items()}
VERB_FRAME = {e["sense"]: e["frame"]
              for es in ENTRIES.values() for e in es if e["pos"] == "V"}
VOPT = {e["sense"]: e["opt"]
        for es in ENTRIES.values() for e in es if e["pos"] == "V"}
SENSE_FORM = {e["sense"]: f
              for f, es in ENTRIES.items() for e in es if e["pos"] == "V"}
PROP_KEYS = ("animate", "edible", "liquid", "place", "readable", "sport")

NUMERALS = set("一二三四五六七八九十两")
CLASSIFIERS = {f for f, es in ENTRIES.items() if any(e["pos"] == "CL" for e in es)}


def entity_info(lex, head):
    """解析角色填充词的信息。

    支持三种 head：
      - 普通词形（"苹果"）
      - 并列复合（"我+小明"）：属性取各部分合取（all），词类拼成 "HUMAN+HUMAN"
      - 数量短语（"一个苹果"）：剥掉"数词+量词"前缀，取基础名词的属性
    返回 {"props":..., "cls":...} 或 None。
    """
    if "+" in head:
        parts = head.split("+")
        infos = [entity_info(lex, p) for p in parts]
        if any(i is None for i in infos):
            return None
        props = {k: all(i["props"][k] for i in infos) for k in PROP_KEYS}
        return {"props": props, "cls": "+".join(i["cls"] for i in infos)}
    info = lex.get(head)
    if info is not None and "props" in info:
        return info
    if len(head) >= 3 and head[0] in NUMERALS and head[1] in CLASSIFIERS:
        info = lex.get(head[2:])
        if info is not None and "props" in info:
            return info
    return None
