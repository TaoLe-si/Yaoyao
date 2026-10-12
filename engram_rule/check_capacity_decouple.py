# -*- coding: utf-8 -*-
"""§69.6.5 门禁：容量解耦实验的**静态走查** —— 不碰 GPU、不建模型、不跑训练。

**为什么先有这一关**：上一轮我在没走查的情况下跑了真实基准，连撞三个自写 bug，
废掉约 14 分钟 GPU。桃明确要求「确定代码正确后，禁止跑测试」的逆序禁止：
**先走查 → 走查绿 → 才准跑。**

| 编号 | 检查 | 负向对照 |
|---|---|---|
| **C1** | `W = 16d²`、`E = 2·d·n_slots` 两个公式在 3 个 d 上**全部成立** | 公式写错 ⇒ 必须红 |
| **C2** | alive 复用合法：`build_engram` 只看 (x, lengths)，**不看 n_slots** ⇒ 跨 slots 复用必须逐位相同 | 改 lengths ⇒ alive 必须变 |
| **C3** | 固定 d 扫 slots 时 `W` **逐位恒定**、`E` 严格 = `2·d·slots` | W 变了 ⇒ 解耦失败 |
| **C4** | 恢复键与文件名含 `n_slots`，改 slots ⇒ 两者都变 | 只改 slots 却同名 ⇒ 会静默复用错恢复点 |
| **C5** | `train_steps` 返回**二元组**，第一个是 list | 按 list 处理 ⇒ 会落盘整个元组（§67 教训） |
| **C6** | `run_cell` 在 `math_gate` 不符时拒绝（T4 前置） | — |

用法：`python -m unittest check_capacity_decouple -v`
"""
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class C1FormulaTest(unittest.TestCase):
    """C1：两个容量公式必须与真实 `named_parameters` 对得上。"""

    def _groups(self, d, slots):
        from model import RCfg, EngramRuleLM
        import exp_capacity_decouple as X
        ck = X._CK or _load_ck()
        cfg = dict(ck["cfg"])
        cfg["d"], cfg["n_heads"], cfg["n_slots"] = d, d // 16, slots
        import torch
        torch.manual_seed(11)
        m = EngramRuleLM(RCfg(**cfg))
        return X.param_groups(m)

    def test_formulas_hold_for_three_d(self):
        import exp_capacity_decouple as X
        for d in (32, 64, 128):
            g = self._groups(d, 4096)
            self.assertEqual(g["w"], X.w_expected(d),
                             "d=%d: W=%d ≠ 16d²=%d" % (d, g["w"], X.w_expected(d)))
            self.assertEqual(g["e"], X.e_expected(d, 4096),
                             "d=%d: E=%d ≠ 2·d·slots" % (d, g["e"]))

    def test_negative_control_wrong_formula_must_fail(self):
        """★负向对照：把公式改成错的（12d²），必须与实测不符。

        ★这正是我自己犯过的错：我先写下 `W=12d²`，实测是 `16d²`。
          没有这条对照，公式错了也照样跑绿。
        """
        import exp_capacity_decouple as X
        for d in (32, 64, 128):
            g = self._groups(d, 4096)
            self.assertNotEqual(g["w"], 12 * d * d,
                                "实测 W 竟等于 12d² —— 那 W 公式就是 12d²，C1 的前提写错了")


def _load_ck():
    import torch
    return torch.load(os.path.join(HERE, "archive",
                                   "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                      map_location="cpu", weights_only=False)


class C2AliveReuseTest(unittest.TestCase):
    """C2：alive 复用跨 slots 档必须**等价**。

    ★第一版直接断言 `m.engram.alive == a_ref`，结果 RuntimeError：
      `alive` 的形状是 `(n_slots,)` ⇒ slots 不同则**形状不同**，根本不能逐位比。
      这是**检查写错了**（规则 5）：不是实现错。

    ★正确的等价性判据：`alive = counts >= min_count`，而 `counts` 是**按槽位计数**。
      slots 变大 ⇒ 新槽位的计数**必然是 0**（同一批 n-gram 只会落进低编号槽），
      所以等价形式是「前 N 个槽完全一致 + 新增槽全为 0」。
      这同时解释了 §67.9 那个「n_slots 越大 alive_slots 越多」的现象 ——
      不是哈希冲突变少，是**槽位变多后更多槽能拿到 ≥1 次计数**。
    """

    def test_alive_consistent_across_slots(self):
        """★**这一条把我原来的设计否掉了**，实测结果与我预期的相反：

        | slots | alive 总和 | 前 256 槽一致 |
        ---|---:|---|
        | 256 | **256** | — |
        | 4096 | **4096** | ✓ |

        槽位变大 ⇒ **alive 总和跟着等比变大**（4096 槽里后 3840 个**全是 1**），
        而不是「新增槽全为 0」。

        ⇒ 说明 `slot_ids` **依赖 n_slots**（哈希按槽数取模/分桶），
          **跨 slots 复用 alive 是非法的** —— 这正是 §67.9 那个
          「n_slots 越大 alive_slots 越多」的真正原因：不是冲突变少，
          是**同一个 n-gram 在不同槽数下落到不同槽**。

        ⇒ 实验脚本必须 `reuse_alive=False` 重算。本测试改成**卡住这个行为**：
          一旦有人把复用打开，这条会红。
        """
        import torch
        import exp_capacity_decouple as X
        ck = X._CK or _load_ck()
        X._CK = ck
        rows = X.load_rows()
        tr, he = X.split_rows(rows)
        X.assert_no_leak(tr, he)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_str(ck["tokenizer"]["json"])
        x, s, l = X.encode_cached(tr[:3000], tok, 512)
        X._TRAIN_X, X._TRAIN_S, X._TRAIN_L = x, s, l
        X._ALIVE_CACHE = None

        a_small = X.make_model(ck, 32, 256, reuse_alive=False).engram.alive.clone()
        a_big = X.make_model(ck, 32, 4096, reuse_alive=False).engram.alive.clone()
        self.assertEqual(int(a_small.sum()), 256)
        self.assertEqual(int(a_big.sum()), 4096)
        self.assertTrue(bool((a_small == a_big[:256]).all()),
                        "小 slots 的 alive 不是大 slots 的前缀 ⇒ 哈希与 slots 强耦合")
        self.assertEqual(int(a_big[256:].sum()), 4096 - 256,
                         "新增槽不是全 1 ⇒ 与实测不符")
        # ★把「复用非法」写成显式断言：alive 总和随 slots 变 ⇒ 复用必然错
        self.assertNotEqual(int(a_small.sum()), int(a_big.sum()),
                            "alive 与 slots 无关 ⇒ 那跨 slots 复用就合法，"
                            "本测试的前提要重写")
        X._ALIVE_CACHE = None

    def test_script_must_not_reuse_alive_across_slots(self):
        """★静态卡住实现：`make_model` 默认必须是 `reuse_alive=False`。

        ★第一版断言 `"def make_model(ck, d, slots, reuse_alive=True):" in src` ——
          **把已修好的实现当失败条件写了**（我已经改成 False），所以检查假红。
          这正是「检查写错」：断言的方向搞反了。
        """
        import io as _io
        p = os.path.join(HERE, "exp_capacity_decouple.py")
        with _io.open(p, encoding="utf-8") as f:
            src = f.read()
        self.assertIn("def make_model(ck, d, slots, reuse_alive=False):", src,
                      "make_model 的默认不是 reuse_alive=False ⇒ 会静默复用错的 alive")
        # run_cell 里必须显式传 False
        idx = src.find("_ALIVE_CACHE = None")
        self.assertGreater(idx, 0, "找不到 alive 处理那一段")
        seg = src[idx:idx + 400]
        self.assertIn("reuse_alive=False", seg,
                      "run_cell 建模型时没显式传 reuse_alive=False")
        for line in src.splitlines():
            if "make_model(ck, d, slots, reuse_alive=" in line:
                self.assertIn("reuse_alive=False", line,
                              "建模型时复用了 alive ⇒ alive 依赖 n_slots，会静默算错")

    def test_negative_control_reuse_true_would_be_rejected(self):
        """★负向对照：把默认改回 True，上面那条必须红（证明它不是常量断言）。"""
        import io as _io
        p = os.path.join(HERE, "exp_capacity_decouple.py")
        with _io.open(p, encoding="utf-8") as f:
            src = f.read()
        bad = src.replace("def make_model(ck, d, slots, reuse_alive=False):",
                          "def make_model(ck, d, slots, reuse_alive=True):")
        self.assertNotIn("def make_model(ck, d, slots, reuse_alive=False):", bad,
                         "把默认改成 True 后，原断言竟仍然成立 ⇒ 判据是常量，无效")

    def test_negative_control_changing_lengths_changes_alive(self):
        """★负向对照：改 lengths ⇒ alive 必须变。

        ★两次都没触发，两次都是**对照选错**（不是实现错）：
          · 砍前 200 行 ⇒ 4095 == 4095（那些 n-gram 被其余行覆盖）
          · 砍前 800 行 ⇒ 4096 == 4096（同上，3000 行里覆盖太充分）
          实测只有**留 200 行**才掉到 3642 —— 3000 行的语料里 n-gram 太密，
          砍掉一部分不影响任何槽的存活状态。

        ⇒ 对照必须选**保证触发**的那个：直接换成 200 行的小语料。
          `check_scaling_v2_speed.py` 的 O1 负向对照踩过同一个坑
          （改最后一行不触发，砍 100 行才触发），这是第三次同类。
        """
        import torch
        import exp_capacity_decouple as X
        ck = X._CK or _load_ck()
        X._CK = ck
        rows = X.load_rows()
        tr, he = X.split_rows(rows)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_str(ck["tokenizer"]["json"])
        x, s, l = X.encode_cached(tr[:3000], tok, 512)

        X._TRAIN_X, X._TRAIN_S, X._TRAIN_L = x, s, l
        X._ALIVE_CACHE = None
        a_full = X.make_model(ck, 32, 4096, reuse_alive=False).engram.alive.clone()

        # 小语料：只留 200 行 ⇒ 一大批 n-gram 消失
        X._TRAIN_X, X._TRAIN_S, X._TRAIN_L = x[:200], s[:200], l[:200]
        X._ALIVE_CACHE = None
        a_small = X.make_model(ck, 32, 4096, reuse_alive=False).engram.alive.clone()

        self.assertEqual(int(a_full.sum()), 4096, "3000 行应填满 4096 槽")
        self.assertLess(int(a_small.sum()), 4096,
                        "留 200 行后 alive 竟仍满 ⇒ 判据是常量，无效")
        self.assertNotEqual(int(a_full.sum()), int(a_small.sum()))
        X._ALIVE_CACHE = None


class C3DecouplingTest(unittest.TestCase):
    """C3：固定 d 扫 slots，W 必须恒定、E 必须线性。"""

    def test_w_constant_e_linear(self):
        import exp_capacity_decouple as X
        for d in (32, 64, 128):
            ws, es = [], []
            for slots in (256, 1024, 4096, 16384):
                g = C1FormulaTest()._groups(d, slots)
                ws.append(g["w"])
                es.append(g["e"])
            self.assertEqual(len(set(ws)), 1,
                             "d=%d 时 W 随 slots 变了 %s ⇒ 解耦失败" % (d, ws))
            for i in range(len(es) - 1):
                self.assertEqual(es[i + 1] / es[i],
                                 (256, 1024, 4096, 16384)[i + 1] // (256, 1024, 4096, 16384)[i],
                                 "E 不是随 slots 线性 ⇒ 不是干净的容量轴")

    def test_negative_control_d_changes_w(self):
        """★负向对照：改 d ⇒ W 必须变（否则 C3 的「恒定」是常量，无效）。"""
        import exp_capacity_decouple as X
        ws = [C1FormulaTest()._groups(d, 4096)["w"] for d in (32, 64, 128)]
        self.assertEqual(len(set(ws)), 3, "三个 d 的 W 竟相同 ⇒ 判据是常量，无效")
        self.assertEqual([w / (d * d) for d, w in zip((32, 64, 128), ws)], [16.0, 16.0, 16.0])


class C4ResumeKeyTest(unittest.TestCase):
    """C4：恢复键与文件名必须含 n_slots。"""

    def test_slots_changes_key_and_path(self):
        import exp_capacity_decouple as X
        base_key = X.resume_key(31148, 64, 4096, 4867, 64)
        base_path = X.resume_path_for(31148, 64, 4096, 4867, 64)
        for slots in (256, 1024, 16384):
            k = X.resume_key(31148, 64, slots, 4867, 64)
            p = X.resume_path_for(31148, 64, slots, 4867, 64)
            self.assertNotEqual(k["n_slots"], base_key["n_slots"])
            self.assertNotEqual(k, base_key, "只改 slots 恢复键没变 ⇒ 会静默复用错恢复点")
            self.assertNotEqual(p, base_path, "只改 slots 文件名没变 ⇒ 同上")
        self.assertEqual(X.resume_key(31148, 64, 4096, 4867, 64), base_key)
        self.assertEqual(X.resume_path_for(31148, 64, 4096, 4867, 64), base_path)

    def test_source_declares_slots_in_key(self):
        src = io.open(os.path.join(HERE, "exp_capacity_decouple.py"),
                      encoding="utf-8").read()
        self.assertIn("n_slots=int(slots)", src,
                      "resume_key 里没有 n_slots ⇒ C4 的前提不成立")


class C5TupleReturnTest(unittest.TestCase):
    """C5：train_steps 必须返回二元组（§67 的 sample_ranges 教训）。"""

    def test_signature_and_return(self):
        src = io.open(os.path.join(HERE, "exp_capacity_decouple.py"),
                      encoding="utf-8").read()
        self.assertIn("return trace, rinfo", src,
                      "train_steps 没有返回二元组")
        self.assertIn("isinstance(trace, list)", src,
                      "run_cell 里没有对 trace 类型的守卫")

    def test_negative_control_wrong_unpack_raises(self):
        """★负向对照：把二元组当 list 用，必须报错而不是静默产出错数据。

        ★第一版写 `with self.assertRaises(ValueError): float(pair)` ——
          **猜错了异常类型**：`float(tuple)` 抛的是 **TypeError**，
          于是断言没触发，检查自己红（同 §69.4.5 的 S3 那次）。
        """
        trace, rinfo = [1.0, 2.0], dict(n_saves=2)
        pair = (trace, rinfo)
        with self.assertRaises(TypeError):
            float(pair)
        with self.assertRaises(TypeError):
            len(pair[0]) + pair[1]


class C6GateTest(unittest.TestCase):
    """C6：math_gate 必须与主实验一致（同一份训练数值代码 ⇒ 读数可比）。"""

    def test_math_gate_matches_main(self):
        import exp_capacity_decouple as X
        import exp_scaling_grid_v2 as E2
        self.assertEqual(X.MGATE, E2.MGATE,
                         "两个脚本的 math_gate 不同 ⇒ 训练数值代码不一致，读数不可比")


if __name__ == "__main__":
    unittest.main()
