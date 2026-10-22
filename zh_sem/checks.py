# -*- coding: utf-8 -*-
"""静态检查 v0.1。每条都必须能在反例下变红（专家评审 A7）。"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import HELDOUT, TRAIN, TRAIN_INVALID  # noqa: E402
from learner import cand_feats, randomize_props  # noqa: E402
from lexicon import LEX, PROP_KEYS, VERB_FRAME  # noqa: E402
from parser import parse, tokenize  # noqa: E402
from semantics import canon  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)


def main():
    # C1 分词往返
    bad = [s for s, _ in TRAIN + HELDOUT if "".join(tokenize(s)) != s]
    check("C1 分词往返", not bad, "bad=%s" % bad[:3])

    # C2 正例必须都有 gold 候选（含 entity）
    bad = []
    for sent, gold in TRAIN + HELDOUT:
        if gold is None:
            continue
        if not any(canon(m) == canon(gold) for m in parse(sent)):
            bad.append((sent, [str(canon(m)) for m in parse(sent)]))
    check("C2 正例可解析", not bad, "bad=%s" % bad[:2])

    # C3 解析失败必须精确等于预期集合（不是恒真）
    no_cand = {s for s, _ in TRAIN + HELDOUT if not parse(s)}
    expected = {"猫给狗", "苹果很跑"}
    check("C3 解析失败集合精确", no_cand == expected,
          "got=%s expected=%s" % (sorted(no_cand), sorted(expected)))

    # C4 train/heldout 不重叠
    tr, hd = {s for s, _ in TRAIN}, {s for s, _ in HELDOUT}
    check("C4 train/heldout 不重叠", not (tr & hd), "overlap=%s" % (tr & hd))

    # C5 属性置换真的改变了属性向量与属性列
    lex2, changed = randomize_props(LEX, seed=7)
    nouns = [f for f, i in LEX.items() if i["pos"] in ("N", "PRON")]
    frac = changed / max(1, len(nouns))
    col_changed = {k: sum(1 for f in nouns if lex2[f]["props"][k] != LEX[f]["props"][k])
                   for k in PROP_KEYS}
    cells_moved = sum(col_changed.values())
    cols_touched = sum(1 for v in col_changed.values() if v > 0)
    check("C5 属性置换有效",
          frac >= 0.70 and cells_moved >= len(nouns) * len(PROP_KEYS) * 0.15 and cols_touched >= 2,
          "changed=%d/%d (%.2f) cells=%d/%d cols_touched=%d %s"
          % (changed, len(nouns), frac, cells_moved,
             len(nouns) * len(PROP_KEYS), cols_touched, col_changed))

    # C6 特征确定性
    m = parse("我吃饭")[0]
    check("C6 特征确定性", cand_feats(m, LEX) == cand_feats(m, LEX))

    # C7 position_only 只含 role 特征
    pos_fs = cand_feats(m, LEX, mode="position_only")
    check("C7 position_only 不含动词/填充词",
          all(f[0] == "role" for f in pos_fs), "fs=%s" % (pos_fs[:3],))

    # C8 词条完整性：名词四属性齐全、动词有 frame
    bad = []
    for f, i in LEX.items():
        if i["pos"] in ("N", "PRON"):
            p = i.get("props")
            if not p or any(k not in p for k in PROP_KEYS):
                bad.append(f)
        if i["pos"] == "V" and not i.get("frame"):
            bad.append(f)
    check("C8 词条完整性", not bad, "bad=%s" % bad)

    # C9 负例要么有候选（交给打分器），要么在预期解析失败集合里
    bad = [s for s, g in TRAIN_INVALID
           if g is None and not parse(s) and s not in expected]
    check("C9 负例无候选仅限预期集合", not bad, "bad=%s" % bad)

    # C10 大冻结 held-out 的构式覆盖（每类 >=15 条）
    p = os.path.join(HERE, "large_heldout.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            large = json.load(f)
        counts = Counter(r["tag"] for r in large)
        check("C10 大 held-out 构式覆盖",
              len(counts) >= 10 and min(counts.values()) >= 15,
              "%d tags, min=%d" % (len(counts), min(counts.values())))
    else:
        check("C10 大 held-out 构式覆盖", False, "large_heldout.json 缺失")

    print("\n%s" % ("全部通过" if not FAILS else "失败: %s" % FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
