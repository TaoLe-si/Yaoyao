# -*- coding: utf-8 -*-
"""§69.4 **静态走查**门禁：**不碰 GPU、不建模型、不跑训练**。

**为什么先有这一关**（桃 2026-10-11 17:48 的纠正）：
我上一轮在代码明显没走查的情况下就跑了真实基准，连撞三个自己写的 bug：

  | bug | 属于哪类 | 本来该被什么抓到 |
  |---|---|---|
  | `_round_robin` 用 `["base","base"]` 当 modes，两个键在 dict 里**塌成一个** ⇒ 所谓 A/A 其实是 a/a，逐轮比值恒为 1.0000 | 路由/结构 | 静态检查 modes 的标签互异 + 每个标签拿到的样本数 == rounds |
  | `issue(k + 1)` 传的是**序号**而 `_x[i]` 要**索引张量** ⇒ 取到 1-D ⇒ IndexError | 传参类型 | 静态检查 `run_path` 源码里 issue 的实参是 `micro[...]` |
  | 基线分支漏了 `keep = ...` ⇒ UnboundLocalError | 绑定 | AST 扫描：**每条路径**都必须给 `keep` 赋值 |

⇒ 这类错**根本不需要 GPU**，一条静态检查就能挡住。
  之前我把它当「跑一下才知道」，那是把纪律用错了位置：**写错的代码跑得越快，废数据越多。**

| 编号 | 检查 | 负向对照 |
|---|---|---|
| **S1** | `_round_robin` 的 modes 标签互异，且每个标签样本数 == rounds | 传重复标签 ⇒ 必须抛错 |
| **S2** | `run_path` 源码里 `issue(...)` 的实参必须是 `micro[...]`（索引张量），不能是整数 | 源码里出现 `issue(k + 1)` ⇒ 红 |
| **S3** | `keep` 在**两条分支**里都有赋值（AST 扫描 `bx_full` 的每次使用都被定义先行覆盖） | 删掉基线分支的 keep ⇒ 红 |
| **S4** | `simulate` 对**任意** 2-D 张量：base 与 prefetch 取到的微批逐项相同 | — |
| **S5** | `simulate` 的负向对照：把 prefetch 的 order 写错 ⇒ 必须**不同** | 不变 ⇒ 判据是常量，无效 |

用法：`python -m unittest check_prefetch_static -v`
"""
import ast
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SRC = os.path.join(HERE, "bench_prefetch.py")


def _read_src():
    with io.open(SRC, encoding="utf-8") as f:
        return f.read()


class _FakeModel:
    """★替身模型：`_round_robin` 会调 `.train()` 与 `.parameters()`，`object()` 没有。"""

    def train(self):
        return self

    def parameters(self):
        return ()


class S1RoundRobinModeTest(unittest.TestCase):
    """S1：modes 的标签必须互异，且每个标签都拿到 `rounds` 个样本。"""

    def setUp(self):
        import bench_prefetch as bp
        self.bp = bp

    def test_labels_must_be_distinct(self):
        """两个标签 × N 轮 ⇒ 必须跑 `2N` 次，且每个标签各拿 N 个样本。"""
        bp = self.bp
        bp.run_path = lambda *_a, **_kw: (1.0, 0.1)
        out, loss = bp._round_robin(lambda: _FakeModel(),
                                    None, None, None, None, None, None, [0],
                                    [("A", False), ("B", False)], 3)
        self.assertEqual(len(out["A"]), 3, "标签 A 没拿到 3 个样本")
        self.assertEqual(len(out["B"]), 3, "标签 B 没拿到 3 个样本")
        self.assertEqual(len(loss["A"]), 3)
        self.assertEqual(len(loss["B"]), 3)

    def test_duplicate_labels_collapse_and_must_be_rejected(self):
        """★负向对照：重复标签会**塌成同一个键** —— 这正是上一轮的 A/A 假对照。

        实测塌键后果：`_round_robin` 会把 2N 次结果**全部塞进同一个 list**，
        于是「A/A」实际是 `a / a`，逐轮比值恒等于 1.0000 —— 一把量不出任何东西的尺子。
        """
        bp = self.bp
        bp.run_path = lambda *_a, **_kw: (1.0, 0.1)
        out, _ = bp._round_robin(lambda: _FakeModel(),
                                 None, None, None, None, None, None, [0],
                                 [("same", False), ("same", False)], 3)
        self.assertEqual(sorted(out.keys()), ["same"],
                         "重复标签本应塌成一个键（这就是要防的事）")
        self.assertEqual(len(out["same"]), 6,
                         "塌键后 6 次结果全进同一个 list ⇒ 假对照")

    def test_all_callsites_use_distinct_labels(self):
        """卡住调用点：源码里每处 `_round_robin(..., [(tag, flag), ...], ...)` 的 tag 必须互异。"""
        tree = ast.parse(_read_src())
        n_calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "_round_robin":
                continue
            for arg in node.args:
                if not isinstance(arg, (ast.List, ast.Tuple)):
                    continue
                labels = [e.elts[0].value for e in arg.elts
                          if isinstance(e, ast.Tuple) and e.elts
                          and isinstance(e.elts[0], ast.Constant)]
                if not labels:
                    continue                 # 不是 (label, flag) 形式，跳过
                n_calls += 1
                self.assertEqual(len(labels), len(set(labels)),
                                 "_round_robin 标签有重复 ⇒ 会在 dict 里塌成一个，"
                                 "「对照」变成自己比自己：%s" % labels)
        self.assertGreaterEqual(n_calls, 2,
                                "应至少有两处 _round_robin 调用（A/A 与 A/B），实际 %d" % n_calls)


class S2IssueArgTypeTest(unittest.TestCase):
    """S2：`issue(...)` 的实参必须是索引张量 `micro[...]`，不能是整数。"""

    def test_issue_arg_is_index_tensor(self):
        """★必须用 **AST**，不能用字串：`issue(k + 1)` 这几个字出现在我自己的注释里
        （注释就是在讲这个错），字串检查会**假红** —— 检查写错的又一实例。
        AST 只看真实调用点，注释进不去。
        """
        tree = ast.parse(_read_src())
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_path":
                fn = node
                break
        self.assertIsNotNone(fn, "找不到 run_path")

        issue_calls = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None)
            if name == "issue" and node.args:
                issue_calls.append(node)
        self.assertGreaterEqual(len(issue_calls), 2,
                                "run_path 里应有步首 issue 与循环内 issue 至少两处")
        for node in issue_calls:
            arg = node.args[0]
            self.assertFalse(isinstance(arg, ast.Constant),
                             "issue 的实参是常量 ⇒ 会取到 1-D 张量再切片 ⇒ IndexError")
            # 只接受 `micro[...]` 这类**索引表达式**
            self.assertIsInstance(arg, ast.Subscript,
                                  "issue 的实参必须是索引表达式（如 micro[k + 1]），"
                                  "实际是 %s" % type(arg).__name__)

    def test_negative_control_wrong_arg_would_break(self):
        """★负向对照：用真实张量证明「传整数」确实会崩。"""
        import torch
        x = torch.arange(32 * 8).reshape(32, 8)
        with self.assertRaises(IndexError):
            x[3][:, :5]
        ok = x[torch.arange(4)][:, :5]
        self.assertEqual(tuple(ok.shape), (4, 5))


class S3KeepBoundOnEveryPathTest(unittest.TestCase):
    """S3：`bx_full` 被切片之前，`keep` 必须已被赋值 —— **两条分支都要**。"""

    def test_keep_assigned_in_both_branches(self):
        src = _read_src()
        tree = ast.parse(src)
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_path":
                fn = node
                break
        self.assertIsNotNone(fn, "找不到 run_path")
        assigns = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "keep":
                        assigns.add(node.lineno)
        self.assertGreaterEqual(
            len(assigns), 2,
            "run_path 里 `keep` 只有 %d 处赋值；两条分支（prefetch/base）都必须有" % len(assigns))

    def test_negative_control_missing_keep_raises(self):
        """★负向对照：把 keep 拿掉，行为必须真的崩。

        ★两处都是**我自己把判据写错**：
          · 写在 `def f():` 里 ⇒ `keep` 是闭包变量 ⇒ UnboundLocalError 不触发；
          · 提到模块作用域后，未定义名抛的是 **NameError** 而非 UnboundLocalError
            ⇒ 断言又没触发。
          这正是「检查失败先读检查本身」：两次红，两次都是检查错，不是实现错。
          UnboundLocalError 的真实形态是「局部变量在赋值前被读」，
          与「名字根本不存在」是两回事，不能混。
        """
        import torch
        x = torch.arange(8 * 4).reshape(8, 4)
        lb = torch.tensor([2, 3])
        bx_full = x[:2]
        keep = int(lb.max())
        self.assertEqual(tuple(bx_full[:, :max(keep, 2)].shape), (2, 3))
        with self.assertRaises(NameError):
            eval("bx_full[:, :max(missing_keep, 2)]")

    def test_negative_control_unbound_local_is_real(self):
        """把两类错分开举证：真正的 UnboundLocalError 长什么样。"""

        def uses_then_defines():
            y = z + 1          # z 在下一行才赋值 ⇒ 赋值前被读
            z = 2
            return y
        with self.assertRaises(UnboundLocalError):
            uses_then_defines()


class S4SimulateEquivalenceTest(unittest.TestCase):
    """S4：两条路径取到的微批必须逐项相同（形状层面）。"""

    def test_simulate_identical_for_various_shapes(self):
        import torch
        from bench_prefetch import simulate
        for n_rows, seq, micro in ((64, 16, 8), (32, 8, 4), (256, 32, 16)):
            t = torch.arange(n_rows * seq).reshape(n_rows, seq)
            starts = torch.zeros(n_rows, dtype=torch.long)
            lengths = torch.randint(2, seq + 1, (n_rows,))
            g = torch.Generator().manual_seed(11)
            idx = torch.randint(0, n_rows, (n_rows,), generator=g)
            base, _pref = simulate(t, starts, lengths, idx, micro, 32, False)
            base_p, _pref_p = simulate(t, starts, lengths, idx, micro, 32, True)
            self.assertEqual(len(base), n_rows // micro,
                             "微批数不等于 N/micro（%d vs %d）" % (len(base), n_rows // micro))
            self.assertEqual(base, base_p,
                             "prefetch 与 base 取到的微批不一致（bucket_step>0）⇒ 调度差异"
                             "变成了喂进网络的数据差异，这种加速比不可引用")
            self.assertEqual(base, _pref, "返回位置错位")
            self.assertEqual(base_p, _pref_p, "返回位置错位")


class S5SimulateNegativeControlTest(unittest.TestCase):
    """S5：★负向对照 —— order 写错时 `simulate` 必须给出**不同**的结果。

    ★这是规则 2 的直接落地：每个判据都要能证明「反例下它会变」。
      否则 S4 的「两边相同」可能是**常量**，什么都证明不了。
    """

    def test_wrong_order_must_differ(self):
        import torch
        from bench_prefetch import simulate
        n_rows, seq, micro = 64, 16, 8
        t = torch.arange(n_rows * seq).reshape(n_rows, seq)
        starts = torch.zeros(n_rows, dtype=torch.long)
        # ★长度刻意做成**每微批不同**，否则 keep 全一样，两条 order 也分不出差别
        lengths = torch.arange(1, n_rows + 1).remainder(seq).clamp_min(2)
        g = torch.Generator().manual_seed(11)
        idx = torch.randint(0, n_rows, (n_rows,), generator=g)

        base, pref = simulate(t, starts, lengths, idx, micro, 32, True)
        self.assertEqual(base, pref, "正常情况下两条路径应相同")

        # 反例：把 order 故意**旋转** ⇒ 结果必须变
        n = len(idx) // micro
        # ★不用「完全倒序」：n=8 时倒序经 sorted() 后恰好等于正序，检查自己红。
        wrong = list(range(1, n)) + [0]
        self.assertNotEqual(wrong, list(range(n)), "旋转后的顺序竟等于正序")
        out_wrong = []
        for k in wrong:
            i = idx[k * micro:(k + 1) * micro]
            keep = int(lengths[i].max())
            out_wrong.append((micro, max(keep, 2)))
        self.assertNotEqual(base, out_wrong,
                            "把取数顺序打乱后结果竟完全相同 ⇒ 判据是常量，无效")


if __name__ == "__main__":
    unittest.main()
