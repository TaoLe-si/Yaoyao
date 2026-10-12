# -*- coding: utf-8 -*-
"""§69.23 静态走查：`answer_loss_sums` 让 think 段进 loss —— **改之前先记账**。

**为什么先静态**：这是要动 `dist/dialogue_small.py`（出货路径的离线副本），
桃说过「大的架构改动再批准」。这一件是**纯逻辑改动**（只动 loss mask 边界），
但规矩不变：**先走查、走查绿、才准改。**

── 三件必须先算清的事 ────────────────────────────────────────────────
| 编号 | 问题 | 为什么必须先算 |
|---|---|---|
| **A1** | 改完后 `answer_loss_sums` 的签名怎么变 | 它是出货路径函数，签名变了会波及所有调用方 |
| **A2** | think 段的 loss 权重怎么定 | 直接等权会让 think 段淹没 answer 段（think 中位 54 字符 vs answer 45 token） |
| **A3** | `first_token_weight` 与 think 段怎么共存 | 现有归一化是 `answer_tokens + (w-1)*n`，加 think 后分母要跟着变 |

── 四个可证伪预测 ────────────────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **L1** | 不改动时（`think_start=None`）行为**逐位相同** | 否则不是向后兼容 |
| **L2** | 给 `think_start` 后，think 段 mask 必须从 0 变 1 | 改动没生效 |
| **L3** | ★负向对照：`think_start` 给错（超出序列）⇒ 必须抛错 | 越界不报会静默算错 |
| **L4** | 分母必须含 think 的 token 数 | 否则 think 与 answer 的梯度量级不一致 |

用法：`python check_think_loss.py`
"""
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SRC = os.path.join(HERE, "dist", "dialogue_small.py")


def _load():
    """按路径加载 `dist/dialogue_small.py`（主目录那份 import 不了）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ds_thinkloss", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ds_thinkloss"] = mod
    spec.loader.exec_module(mod)
    return mod


class A1SignatureAudit(unittest.TestCase):
    """A1：当前签名与调用方清单（先记账，再决定怎么改）。"""

    def test_current_signature(self):
        import inspect
        m = _load()
        sig = inspect.signature(m.answer_loss_sums)
        print("    当前签名:", sig)
        self.assertIn("starts", sig.parameters)
        self.assertIn("valid", sig.parameters)
        print("    ⇒ 现有参数 (logits, target, valid, starts)，无 think 相关参数")

    def test_callers_count(self):
        """静态数一遍调用方 —— 改签名必须知道会影响谁。"""
        import re
        n = 0
        for name in os.listdir(HERE):
            if not name.endswith(".py"):
                continue
            with io.open(os.path.join(HERE, name), encoding="utf-8") as f:
                src = f.read()
            n += len(re.findall(r"answer_loss_sums\(", src))
        print("    仓库内 `answer_loss_sums(` 出现次数 = %d" % n)
        self.assertGreater(n, 1, "只有 1 处调用 ⇒ 那这次改的影响面比预期小")


class A2WeightDesign(unittest.TestCase):
    """A2：think 与 answer 的长度比（决定要不要加权）。"""

    def test_length_ratio(self):
        """用真实语料量 think vs answer 的 token 比。"""
        import build_cot_corpus as B
        import exp_scaling_grid_v2 as E2
        rows = E2.load_rows()[:3000]
        made, _st = B.build(rows, 3000)
        tl = [len(r["think"]) for r in made]
        al = [len(r["answer"]) for r in made]
        import statistics
        tm, am = statistics.median(tl), statistics.median(al)
        print("    think 字符中位 %.0f / answer 字符中位 %.0f ⇒ 比 %.2f"
              % (tm, am, tm / max(am, 1)))
        print("    ⇒ 等权会让 think 段的梯度贡献约为 answer 的 %.2f 倍" % (tm / max(am, 1)))
        # ★不预设阈值，只把比值记下来供设计用
        self.assertGreater(tm, 0)
        self.assertGreater(am, 0)


class L1BackwardCompat(unittest.TestCase):
    """L1：不改动时必须逐位相同（这是「可以改」的前提）。"""

    def test_baseline_unchanged(self):
        m = _load()
        T, B = 10, 2
        logits = torch.zeros(B, T, 5)
        target = torch.zeros(B, T, dtype=torch.long)
        valid = torch.ones(B, T + 1, dtype=torch.bool)
        starts = torch.tensor([6, 4])
        r1 = m.answer_loss_sums(logits, target, valid, starts)
        r2 = m.answer_loss_sums(logits, target, valid, starts)
        self.assertEqual(float(r1[0]), float(r2[0]), "L1 失败：同输入两次结果不同")
        print("    ⇒ 现有实现确定性 OK，可作为对照基线")


class L2ThinkMaskWouldChange(unittest.TestCase):
    """L2：模拟「加 think_start 参数后」mask 的变化（纯张量推演，不改源文件）。"""

    def test_mask_with_think(self):
        T, B = 10, 1
        target = torch.zeros(B, T, dtype=torch.long)
        valid = torch.ones(B, T + 1, dtype=torch.bool)
        starts = torch.tensor([6])
        think_start = torch.tensor([2])       # think 段从第 2 位开始

        positions = torch.arange(1, target.shape[1] + 1).view(1, -1)
        old = valid[:, 1:] & (positions >= starts.view(-1, 1))
        # 新口径：think 段（think_start ≤ pos < starts）+ answer 段都进 loss
        new = valid[:, 1:] & ((positions >= starts.view(-1, 1))
                              | (positions >= think_start.view(-1, 1)))
        print("    旧 mask:", [int(v) for v in old[0].tolist()])
        print("    新 mask:", [int(v) for v in new[0].tolist()])
        self.assertFalse(any(old[0, :2].tolist()), "旧 mask 前 2 位应为 0")
        self.assertTrue(all(new[0, 2:].tolist()), "新 mask 从 think_start 起应全为 1")
        print("    ⇒ think 段（第 2~5 位）从全 0 变全 1 ⇒ L2 成立")


class L3NegativeControl(unittest.TestCase):
    """L3：负向对照 —— think_start 越界必须抛错（不抛会静默算错）。"""

    def test_out_of_range_think_start_must_raise(self):
        """模拟实现里那道守卫：`think_start > starts` 是逻辑错误（think 不可能在 answer 之后）。"""
        T = 10
        starts, think_start = 6, 8       # think 在 answer 之后 ⇒ 逻辑错
        with self.assertRaises(AssertionError):
            assert think_start < starts, \
                "think_start(%d) >= starts(%d)：think 段不可能在 answer 之后" % (think_start, starts)
        print("    ⇒ think_start=8 > starts=6 ⇒ 守卫抛错 ✓")
        print("    ★这一条必须在真实实现里也存在，否则越界会静默算错")

    def test_valid_order_passes(self):
        starts, think_start = 6, 2
        assert think_start < starts
        print("    ⇒ think_start=2 < starts=6 ⇒ 合法 ✓")


class L4Denominator(unittest.TestCase):
    """L4：分母必须含 think 的 token 数（否则梯度量级不一致）。"""

    def test_denominator_includes_think(self):
        """算一遍：加 think 后分母应增加多少。"""
        T, B = 10, 1
        target = torch.zeros(B, T, dtype=torch.long)
        valid = torch.ones(B, T + 1, dtype=torch.bool)
        starts = torch.tensor([6])
        think_start = torch.tensor([2])
        positions = torch.arange(1, target.shape[1] + 1).view(1, -1)
        think_mask = valid[:, 1:] & (positions >= think_start.view(-1, 1)) \
            & (positions < starts.view(-1, 1))
        n_think = int(think_mask.sum())
        n_answer = int((valid[:, 1:] & (positions >= starts.view(-1, 1))).sum())
        print("    think 段 token 数 = %d，answer 段 = %d" % (n_think, n_answer))
        print("    ⇒ 分母应从 %d 增至 %d（+%.0f%%）"
              % (n_answer, n_answer + n_think, 100.0 * n_think / max(n_answer, 1)))
        self.assertGreater(n_think, 0, "think 段 token 数为 0 ⇒ 判据是常量")
        # 负向对照：think_start == starts 时 think 段必须为空
        ts2 = torch.tensor([6])
        m2 = valid[:, 1:] & (positions >= ts2.view(-1, 1)) & (positions < starts.view(-1, 1))
        self.assertEqual(int(m2.sum()), 0,
                         "think_start == starts 时 think 段竟非空 ⇒ 边界算错")
        print("    ⇒ 负向对照 think_start==starts ⇒ think 段为空 ✓")


if __name__ == "__main__":
    unittest.main()
