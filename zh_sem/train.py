# -*- coding: utf-8 -*-
"""训练入口：反馈学习 → 保存离散权重表。"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import TRAIN, TRAIN_VALID  # noqa: E402
from learner import train, train_acc  # noqa: E402
from lexicon import LEX  # noqa: E402

OUT = os.path.join(HERE, "weights.pkl")


def main():
    w = train(TRAIN, LEX, mode="full", epochs=40, seed=0)
    ok, n = train_acc(w, TRAIN, LEX, mode="full")
    print("train accuracy: %d/%d = %.4f" % (ok, n, ok / n))
    if ok != n:
        print("!! 训练集未完全拟合，先查数据/规则，不要直接看 heldout")
    with open(OUT, "wb") as f:
        pickle.dump(dict(w), f)
    print("saved %s (%d features)" % (OUT, len(w)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
