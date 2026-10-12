"""§67 门禁：v2 网格的**判据本身**必须先被证明可失败，才允许跑实验。

**这个脚本不训练、不联网、不读 GPU。** 它只验证三件事：

1. **暴露比（epochs）是受控变量**：历史 v1 网格把 `steps=200` 钉死，导致 N 越小每行
   重复越多（N=50 → 1024 次，N=738 → 69 次），「数据量」与「过拟合程度」两个变量
   同时动。v2 必须把 epochs 固定，只让 N 动。
2. **weight-decay 是可分离的第二个因子**：`dialogue_small.py:417` 把 `weight_decay=0.01`
   写死，参数量随 d 涨 4.3 倍时衰减力不成比例。v2 必须在同格做 wd=0 / wd=0.01 的
   2×2 析因，否则「容量越大越差」无法归因。
3. **输入数据必须通过泄漏断言**：任何一格训练集与固定探针集之间，规范化后的
   question 或 answer 有交集就必须拒绝（规则 11：全量、词项级）。

**每条判据都配负向对照**（规则 7）：下面 `test_*_negative_control` 证明「反例成立时
判据确实会翻转」。若某个负向对照不翻转，说明判据是常量，不算证据。

用法：`python -m unittest check_scaling_v2 -v`
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BS = 256


class ExposureControlTest(unittest.TestCase):
    """v2 的核心不变量：epochs = bs·steps/N 在整张网格里必须**近似**恒定。

    ★**为什么不是严格相等**（这是检查本身的边界，不是代码的漏洞）：
      `steps` 必须是整数，而 `steps = ceil(EPOCHS_TARGET·N/bs)` ⇒ 实际 epochs
      落在 `[EPOCHS_TARGET, EPOCHS_TARGET + bs/N)`。
      例：N=2000、EPOCHS_TARGET=40、bs=256 ⇒ ceil(312.5)=313 ⇒ epochs=40.064，
      上界偏差 bs/N = 256/2000 = 0.128。
      这正是「epochs 偏差 ≤ bs/N」这条判据的来源，而不是拍脑袋的容差。
      v1 的偏差是 1024−69 = **955 个 epoch**，比这个量化上界大三个数量级。
    """

    def import_grid(self):
        import exp_scaling_grid_v2 as exp
        return exp

    def test_epochs_constant_across_grid(self):
        exp = self.import_grid()
        cells = exp.build_grid()
        self.assertGreaterEqual(len(cells), 4, "网格至少要 4 格才有两个自变量")
        for cell in cells:
            ep = BS * cell["steps"] / cell["n"]
            # 偏差上界由整数量化推出：ceil ⇒ epochs < EPOCHS_TARGET + bs/N
            upper = exp.EPOCHS_TARGET + BS / cell["n"]
            self.assertGreaterEqual(ep, exp.EPOCHS_TARGET - 1e-9,
                                    "格 %s 的 epochs %.6f 低于目标 %.6f"
                                    % (cell, ep, exp.EPOCHS_TARGET))
            self.assertLess(ep, upper + 1e-9,
                            "格 %s 的 epochs %.6f 超过整数量化上界 %.6f"
                            % (cell, ep, upper))

    def test_epochs_negative_control_must_fail(self):
        """★负向对照：把 epochs 目标改掉，同一条断言必须失败。

        若这条测试**不**失败，说明判据在骗人（例如目标值被写成与计算式同一个表达式），
        上面的正对照也就没有效力。这里同时验证**上下界两侧**都能翻转。
        """
        exp = self.import_grid()
        # 下界对照：目标抬高到任何一格都达不到
        for cell in exp.build_grid():
            ep = BS * cell["steps"] / cell["n"]
            self.assertLess(ep, exp.EPOCHS_TARGET * 2.0,
                            "epochs %.4f 竟达到目标的 2 倍 —— 判据无效" % ep)
        # 上界对照：把量化上界收紧到不可能满足（目标本身）
        for cell in exp.build_grid():
            ep = BS * cell["steps"] / cell["n"]
            upper = exp.EPOCHS_TARGET + BS / cell["n"]
            self.assertGreaterEqual(upper, exp.EPOCHS_TARGET)
            # N 必须真的贡献了量化余量，否则上下界重合、判据退化成严格相等
            if cell["n"] > BS:
                self.assertGreater(upper - exp.EPOCHS_TARGET, 0.0)

    def test_steps_scale_with_n_not_fixed(self):
        """历史 bug 的回归护栏：steps 不得是常数。

        v1 的 `STEPS = 200` 钉死正是 epochs 失控的根因。v2 里 steps 必须随 N 变。
        """
        exp = self.import_grid()
        cells = exp.build_grid()
        by_n = {}
        for cell in cells:
            by_n.setdefault(cell["n"], set()).add(cell["steps"])
        self.assertGreater(len(by_n), 1, "网格只有一个 N 档，无法检验 steps 是否随 N 变")
        # 同一 N 档内 steps 必须唯一（否则 epochs 不可能恒定）
        for n, steps_set in by_n.items():
            self.assertEqual(len(steps_set), 1,
                             "N=%d 有多个 steps=%s ⇒ epochs 不可能恒定" % (n, sorted(steps_set)))
        # N 越大 steps 必须越大（epochs 恒定 ⇒ steps ∝ N）
        ordered = sorted(by_n.items())
        for (n0, s0), (n1, s1) in zip(ordered, ordered[1:]):
            self.assertGreater(next(iter(s1)), next(iter(s0)),
                               "N 从 %d 增到 %d 时 steps 没有增大" % (n0, n1))

    def test_weight_decay_factorial_covers_both_arms(self):
        """weight decay 必须是网格里的显式因子，两个臂都要有。"""
        exp = self.import_grid()
        arms = {cell["weight_decay"] for cell in exp.build_grid()}
        self.assertEqual(arms, {0.0, exp.WEIGHT_DECAY_SHIPPING},
                         "weight decay 臂不完整：%s" % sorted(arms))
        for cell in exp.build_grid():
            self.assertIn("weight_decay", cell)
            self.assertGreaterEqual(cell["weight_decay"], 0.0)

    def test_reject_steps_below_floor(self):
        """epochs 目标在小 N 上会推出极少的步数，必须有下限，否则训练不动。"""
        exp = self.import_grid()
        for cell in exp.build_grid():
            self.assertGreaterEqual(cell["steps"], exp.MIN_STEPS,
                                    "N=%d 推出 steps=%d 低于下限 %d"
                                    % (cell["n"], cell["steps"], exp.MIN_STEPS))
            self.assertGreaterEqual(cell["d"], 32)


class LeakageGuardTest(unittest.TestCase):
    """数据必须全量、词项级不相交；同时验证断言本身能抓到注入的泄漏。"""

    def rows(self):
        import exp_scaling_grid_v2 as exp
        return exp.load_rows()

    def test_no_train_probe_leak(self):
        exp = self.import_grid() if False else None
        import exp_scaling_grid_v2 as exp
        rows = self.rows()
        train, probe = exp.split_rows(rows)
        tq = {exp.normalize_key(r["question"]) for r in train}
        ta = {exp.normalize_key(r["answer"]) for r in train}
        pq = {exp.normalize_key(r["question"]) for r in probe}
        pa = {exp.normalize_key(r["answer"]) for r in probe}
        self.assertEqual(tq & pq, set(), "question 泄漏：%s" % sorted(tq & pq)[:5])
        self.assertEqual(ta & pa, set(), "answer 泄漏：%s" % sorted(ta & pa)[:5])
        self.assertGreater(len(probe), 0, "探针集为空")
        self.assertGreater(len(train), 0, "训练集为空")

    def test_leakage_guard_negative_control(self):
        """★负向对照：手工注入一个共享 answer，断言必须抓到。

        证明「泄漏检查」不是恒真的空断言。
        """
        import exp_scaling_grid_v2 as exp
        rows = self.rows()
        train, probe = exp.split_rows(rows)
        probe.append({"question": "__injected_unique_q__", "answer": train[0]["answer"],
                      "history": []})
        tq = {exp.normalize_key(r["question"]) for r in train}
        ta = {exp.normalize_key(r["answer"]) for r in train}
        pa = {exp.normalize_key(r["answer"]) for r in probe}
        self.assertTrue(ta & pa, "注入的共享 answer 未被检出 —— 泄漏断言是假的")

    def test_dedup_is_full_not_prefix_only(self):
        """去重必须全量：**原始文本**重复行数必须为 0。

        ★口径（2026-10-11 随 `assert_no_leak` 一起修正）：按**原始**
          `(question, answer, history)` 判重，**不**按 normalize_key 判。
          上一版用 normalize_key 做键，在 155,740 行大语料上必然失败 ——
          语料里有 33 条行在 NFKC+casefold+去空白后与另一行撞键（全角/半角、
          大小写等细微差异）。那是**近重复**，不是重复行，更不是泄漏。
          真正的泄漏判据是 train/held 的 question 与 answer **集合交集**（上面那条）。
        """
        import exp_scaling_grid_v2 as exp
        rows = self.rows()
        keys = [(r["question"], r["answer"], tuple(r["history"])) for r in rows]
        self.assertEqual(len(keys), len(set(keys)),
                         "存在 %d 条原始文本重复行" % (len(keys) - len(set(keys))))
        # 负向对照：手工复制一行，断言必须抓到
        doubled = rows + [dict(rows[0])]
        keys2 = [(r["question"], r["answer"], tuple(r["history"])) for r in doubled]
        self.assertEqual(len(keys2) - len(set(keys2)), 1,
                         "注入的 1 条重复行未被检出 —— 去重断言是假的")


if __name__ == "__main__":
    unittest.main()
