# -*- coding: utf-8 -*-
"""§69.1 门禁：断点续训必须是**等价续训**，不是「差不多接着跑」。

改动都在 `exp_scaling_grid_v2.py`，**不碰出货路径**（`dialogue_small.py` 一行未改）。

| 编号 | 检查 | 判据 | 负向对照 |
|---|---|---|---|
| **P0** | 确定性基线：同一输入连跑两遍 | max\\|Δ\\| = 0（**P1 的尺子本身**） | 若非 0 ⇒ P1 只能用容差，容差由本读数给出 |
| **P1** | 直跑 12 步 vs 6+存+恢复+6 步 | state_dict 与 P0 同口径，trace 逐点相同 | — |
| **P2** | ★负向对照：恢复时喂**错的** RNG 状态 | 必须**变**，且幅度 > P0 容差 | 不变 ⇒ `gen` 没被用，P1 是假绿 |
| **P3** | 恢复键错配 / 恢复键撞名 | 必须抛错 / 必须不同名 | — |
| **P4** | `resume=None` 不许写任何文件 | 目录文件数前后不变 | — |
| **P5** | 原子写：成功不留 `.tmp`，残留 `.tmp` 不被加载 | — | — |

★**P0 为什么排第一**：本项目吃过三次「判据本身写错」的亏（§67.4 的 epochs 容差、
  §40 的 CUDA event、§63.3 的 LOO 当 CI）。若 GPU 上的 index_add 本身非确定，
  那 P1 的「必须逐位相同」就是一把量不出东西的尺子 —— 必须先量尺子。

用法：`python -m unittest check_resume -v`
"""
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _max_delta(sd_a, sd_b):
    """两个 state_dict 的最大逐元素差；形状/键不同直接返回 inf。"""
    if set(sd_a) != set(sd_b):
        return float("inf")
    worst = 0.0
    for k in sd_a:
        a, b = sd_a[k], sd_b[k]
        if a.shape != b.shape:
            return float("inf")
        if not a.is_floating_point():
            if not bool((a == b).all()):
                return float("inf")
            continue
        worst = max(worst, float((a.double() - b.double()).abs().max()))
    return worst


class _Base(unittest.TestCase):
    """共用一次性的数据/模型准备：build_engram 在 N=2,000 上也要花时间。"""

    @classmethod
    def setUpClass(cls):
        import exp_scaling_grid_v2 as e
        import torch
        cls.e = e
        cls.torch = torch
        ck, tok = e.load_all()
        rows = e.load_rows()
        tr, he = e.split_rows(rows)
        e.assert_no_leak(tr, he)
        x, s, l = e.encode_cached(tr[:2000], tok, e.SEQ)
        e._TRAIN_X, e._TRAIN_S, e._TRAIN_L = x, s, l
        e._CK = ck
        m0 = e.make_model(ck, 32)
        e._ALIVE_CACHE = m0.engram.alive.detach().clone()
        del m0
        if e.DEV == "cuda":
            torch.cuda.empty_cache()
        cls.ck = ck
        cls.pack = (x, s, l)
        cls.tmp = tempfile.mkdtemp(prefix="resume_check_", dir=HERE)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.cell = dict(n=2000, d=32, steps=12, weight_decay=0.0)
        self.mb = 64
        # ★把存盘间隔调到小于总步数，让「被杀」真的落在**训练中途**。
        #   第一版没调，于是 12 < CKPT_EVERY(50) ⇒ 恢复点在**末步**（step=12），
        #   恢复时 start_step >= steps 直接跳过训练 —— 测的根本不是中途恢复。
        #   那是**检查写错了**，不是实现错（规则 5）。
        self._ckpt_every_saved = self.e.CKPT_EVERY
        self.e.CKPT_EVERY = 4
        self.expected_resume_at = 4

    def tearDown(self):
        self.e.CKPT_EVERY = self._ckpt_every_saved

    def _fresh_model(self):
        return self.e.make_model(self.ck, 32, reuse_alive=True)

    def _key(self, **over):
        c = dict(self.cell)
        c.update(over)
        return self.e.resume_key(c, self.mb), c

    def _train(self, model, steps=None, resume=None):
        return self.e.train_steps(model, *self.pack,
                                  steps if steps is not None else self.cell["steps"],
                                  self.mb, self.cell["weight_decay"], resume=resume)

    def _save_then_die(self):
        """把 save_resume 包一层：存完第一次就抛异常，**真实模拟进程被杀**。

        必须在**存盘之后**才抛 —— 崩在存盘之前就没有恢复点可测了；
        而且这正是真实事故的形态：恢复点本身是完好的，是进程没了。
        """
        e = self.e
        orig = e.save_resume
        state = dict(n=0)

        def wrapped(path, key, model, opt, sched, gen, step, trace):
            sec = orig(path, key, model, opt, sched, gen, step, trace)
            state["n"] += 1
            if state["n"] == 1:
                raise KeyboardInterrupt("模拟中断：存完第一次恢复点就杀进程")
            return sec
        return orig, wrapped


class P0DeterminismBaselineTest(_Base):
    """P0：同一输入连跑两遍，先量出「本机 GPU 训练步」的确定性。"""

    def test_two_identical_runs(self):
        ma = self._fresh_model()
        ta, _ = self._train(ma)
        mb_ = self._fresh_model()
        tb, _ = self._train(mb_)
        d = _max_delta(ma.state_dict(), mb_.state_dict())
        self.assertEqual(len(ta), len(tb), "两次跑的 trace 长度不同，尺子本身有问题")
        self.assertEqual(ta, tb, "两次跑的 trace 逐点不同 ⇒ 训练步非确定")
        # 这条断言是**判据**：若本机 index_add 非确定，它会红，红了才允许把 P1 放宽。
        self.assertEqual(d, 0.0,
                         "同一输入两跑的 state_dict max|Δ|=%.3e ≠ 0 ⇒ 本机训练步非确定，"
                         "P1 的「逐位相同」是假判据，必须改用容差" % d)


class P1ResumeEquivalenceTest(_Base):
    """P1：直跑 12 步 == 6 步 + 存 + 恢复 + 6 步。"""

    def _save_then_die(self):
        """见 `_Base._save_then_die`。"""
        return _Base._save_then_die(self)

    def test_resume_equals_uninterrupted(self):
        e = self.e
        # (a) 一路跑满
        m_full = self._fresh_model()
        trace_full, _ = self._train(m_full)

        # (b) 跑一半被杀，再从恢复点接着跑满
        path = os.path.join(self.tmp, "p1.pt")
        key, _cell = self._key()
        m_part = self._fresh_model()
        orig, wrapped = self._save_then_die()
        e.save_resume = wrapped
        try:
            with self.assertRaises(KeyboardInterrupt):
                self._train(m_part, resume=(path, key))
        finally:
            e.save_resume = orig
        self.assertTrue(os.path.exists(path), "被杀之前竟没留下恢复点")
        self.assertEqual(int(self.torch.load(path, map_location="cpu",
                                             weights_only=False)["step"]),
                         self.expected_resume_at,
                         "恢复点的 step 不等于被杀时已完成的步数")

        m_res = self._fresh_model()
        trace_res, rinfo = self._train(m_res, resume=(path, key))
        self.assertEqual(rinfo["resumed_from"], self.expected_resume_at,
                         "恢复后没有从已完成步数接着走")
        self.assertEqual(rinfo["n_saves"], 2, "第二轮应存 2 次（每 CKPT_EVERY + 末步）")

        d = _max_delta(m_full.state_dict(), m_res.state_dict())
        self.assertEqual(d, 0.0,
                         "续训与不中断的 max|Δ|=%.3e ≠ 0 ⇒ 恢复不是等价续训" % d)
        self.assertEqual(trace_full, trace_res,
                         "续训的 loss 轨迹与不中断的不同（%r vs %r）"
                         % (trace_full, trace_res))


class P2NegativeControlTest(_Base):
    """P2：★拆掉 RNG 这根承重墙，恢复结果必须**变**。

    ★这里喂的是**错的** RNG 状态（另 seed 的 generator），不是「不存」——
      因为不存会直接 KeyError 崩掉，崩掉只能证明代码会报错，证明不了
      「RNG 状态是承重墙」。喂错状态才能复现真正危险的场景：
      **静默地**接着跑，但样本序列全变了。
    """

    def _corrupt_gen_save(self, path, key, model, opt, sched, gen, step, trace):
        """存盘时把 `gen` 状态**换成另一个 seed 的**，存完第一次就抛异常。"""
        e = self.e
        os.makedirs(os.path.dirname(path), exist_ok=True)
        wrong = self.torch.Generator().manual_seed(9999).get_state()
        e._atomic_save(dict(key=key, model=model.state_dict(),
                            opt=opt.state_dict(), sched=sched.state_dict(),
                            gen=wrong,               # ★故意给错的
                            torch_rng=self.torch.get_rng_state(),
                            step=int(step), trace=list(trace),
                            math_gate=e.MGATE, saved_at=0.0), path)
        raise KeyboardInterrupt("模拟中断（恢复点里的 RNG 是错的）")

    def test_wrong_rng_state_must_change_result(self):
        e = self.e
        m_full = self._fresh_model()
        trace_full, _ = self._train(m_full)

        path = os.path.join(self.tmp, "p2.pt")
        key, _cell = self._key()
        m_part = self._fresh_model()
        orig = e.save_resume
        e.save_resume = self._corrupt_gen_save          # ★换成写错 RNG 的版本
        try:
            with self.assertRaises(KeyboardInterrupt):
                self._train(m_part, resume=(path, key))
        finally:
            e.save_resume = orig
        self.assertTrue(os.path.exists(path), "P2 没留下恢复点，负向对照没跑起来")

        m_res = self._fresh_model()
        _trace_res, rinfo = self._train(m_res, resume=(path, key))
        self.assertEqual(rinfo["resumed_from"], self.expected_resume_at)
        d = _max_delta(m_full.state_dict(), m_res.state_dict())
        self.assertGreater(d, 0.0,
                           "喂错 RNG 状态后结果竟完全相同 ⇒ gen 没被真正使用，"
                           "P1 的「逐位相同」是假绿（P2 判据失效）")
        self.assertNotEqual(trace_full, _trace_res,
                            "喂错 RNG 状态后 loss 轨迹竟完全相同 ⇒ 同上")


class P3ResumeKeyTest(_Base):
    """P3：恢复键必须覆盖全部自变量，且错配必须抛错。"""

    def test_mismatched_key_raises(self):
        e = self.e
        path = os.path.join(self.tmp, "p3.pt")
        key_a, cell_a = self._key()
        key_b, _cell_b = self._key(n=1280)             # 只改一个自变量
        self.assertNotEqual(key_a, key_b, "只改 n 竟没改变恢复键 ⇒ 恢复键漏了 n")

        m = self._fresh_model()
        # 手工造一个合法恢复点（用与 train_steps 完全相同的构造方式）
        opt = self.torch.optim.AdamW(m.parameters(), lr=e.LR, weight_decay=0.0)
        sched = self.torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=e.LR, total_steps=12, pct_start=0.1)
        gen = self.torch.Generator().manual_seed(e.SEED)
        e.save_resume(path, key_a, m, opt, sched, gen, 3, [1.0])
        del opt, sched, gen

        with self.assertRaises(RuntimeError) as cm:
            e.load_resume(path, key_b)
        self.assertIn("拒绝恢复", str(cm.exception))

    def test_stale_math_gate_raises(self):
        """恢复点的 math_gate 与当前不符 ⇒ 拒绝（训练数值代码已变，权重不可信）。"""
        e = self.e
        path = os.path.join(self.tmp, "p3b.pt")
        key, _c = self._key()
        m = self._fresh_model()
        opt = self.torch.optim.AdamW(m.parameters(), lr=e.LR, weight_decay=0.0)
        sched = self.torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=e.LR, total_steps=12, pct_start=0.1)
        gen = self.torch.Generator().manual_seed(e.SEED)
        e.save_resume(path, key, m, opt, sched, gen, 3, [1.0])
        st = self.torch.load(path, map_location="cpu", weights_only=False)
        st["math_gate"] = "deadbeef0000"
        e._atomic_save(st, path)
        with self.assertRaises(RuntimeError) as cm:
            e.load_resume(path, key)
        self.assertIn("作废", str(cm.exception))

    def test_resume_filename_encodes_all_vars(self):
        """恢复文件名必须自带全部自变量：撞名就意味着可以安全复用。

        ★`cell` 的字段名与 `resume_key` 的键名不是一套（`weight_decay` ↔ `wd`），
          所以这里显式列出映射，不靠 `a[field]` 猜 —— 上一版就是这么写错的。
        """
        e = self.e
        base_path = e.resume_path_for(self.cell, self.mb)
        cases = (("n", "n", 1280), ("d", "d", 64), ("steps", "steps", 200),
                 ("weight_decay", "wd", 0.01))
        for cell_field, key_field, val in cases:
            other = dict(self.cell)
            other[cell_field] = val
            ka = e.resume_key(self.cell, self.mb)
            kb = e.resume_key(other, self.mb)
            self.assertNotEqual(ka[key_field], kb[key_field],
                                "只改 %s 竟没改变恢复键的 %s" % (cell_field, key_field))
            self.assertNotEqual(base_path, e.resume_path_for(other, self.mb),
                                "只改 %s 竟得到同一个恢复文件名 ⇒ 会静默复用错恢复点"
                                % cell_field)
        # micro_batch / n_slots 不在 cell 里，但同样进键与文件名
        self.assertNotEqual(base_path, e.resume_path_for(self.cell, 32),
                            "只改 micro_batch 竟得到同一个恢复文件名")
        self.assertEqual(base_path, e.resume_path_for(self.cell, self.mb),
                         "同一格两次算出的恢复文件名应相同")


class P4NoWriteWhenDisabledTest(_Base):
    """P4：`resume=None`（探针/自测路径）不许写任何文件。"""

    def test_no_resume_writes_nothing(self):
        e = self.e
        probe_dir = os.path.join(self.tmp, "probe")
        os.makedirs(probe_dir, exist_ok=True)
        before = sorted(os.listdir(probe_dir))
        real_dir = e.RESUME_DIR
        try:
            e.RESUME_DIR = probe_dir
            m = self._fresh_model()
            trace, rinfo = self._train(m, steps=3)          # resume=None
        finally:
            e.RESUME_DIR = real_dir
        after = sorted(os.listdir(probe_dir))
        self.assertEqual(before, after,
                         "resume=None 仍写了文件: %s" % sorted(set(after) - set(before)))
        self.assertEqual(rinfo["n_saves"], 0, "resume=None 却报告存了盘")
        self.assertEqual(rinfo["save_wall_sec"], 0.0)
        self.assertEqual(rinfo["resumed_from"], 0)
        self.assertIsInstance(trace, list, "第一个返回值必须是 list（元组陷阱守卫）")


class P5AtomicWriteTest(_Base):
    """P5：原子写 —— 成功不留 `.tmp`；残留 `.tmp` 不会被当成恢复点加载。"""

    def test_atomic_save_leaves_no_tmp(self):
        e = self.e
        path = os.path.join(self.tmp, "p5.pt")
        e._atomic_save({"a": 1}, path)
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path + ".tmp"), "原子写成功后仍留下 .tmp")

    def test_stale_tmp_not_loaded(self):
        """★残留的 `.tmp` 必须被忽略：`train_steps` 只看正式路径。"""
        e = self.e
        path = os.path.join(self.tmp, "p5b.pt")
        with open(path + ".tmp", "wb") as f:
            f.write(b"this is not a torch file")
        m = self._fresh_model()
        trace, rinfo = self._train(m, steps=3, resume=(path, e.resume_key(
            self.cell, self.mb)))
        self.assertEqual(rinfo["resumed_from"], 0,
                         "残留 .tmp 被当成恢复点了")
        self.assertIsInstance(trace, list)


class P6GateEquivalenceTest(unittest.TestCase):
    """P6：`gate_ok` 的等价声明必须**自 policing** —— 训练数值代码一变就自动作废。

    ★这不是小事：等价声明若失效，桃会拿**旧代码跑出来的数**去下结论，
      而那正是本项目反复出现的事故形态（跨轮比较、门禁作废）。

    ★★2026-10-11 大改：原来的 P6 断言 `LEGACY_GATE_EQUIV["math_gate"] == MGATE`
      —— 而那个字段就是用 `math_gate=MGATE` 赋的 ⇒ **这个断言恒真**，
      它在**强制要求**"声明必须自动跟随 MGATE"，也就是在强制要求 §69.28.5 拆掉的那个 bug。
      ⇒ 改成判**真正想要的性质**：
        (a) 声明是**钉死的字面量**，不是运行时算出来的；
        (b) 旧记录确实被接受（可能经由登记过的中性迁移）；
        (c) ★真改了训练数值代码后，旧记录**必须立刻作废**。
    """

    def setUp(self):
        import exp_scaling_grid_v2 as e
        self.e = e
        self.saved_leg = dict(e.LEGACY_GATE_EQUIV)
        self.saved_mig = tuple(dict(m) for m in e.NEUTRAL_MIGRATIONS)
        self.saved_gate = e.MGATE

    def tearDown(self):
        self.e.LEGACY_GATE_EQUIV.clear()
        self.e.LEGACY_GATE_EQUIV.update(self.saved_leg)
        self.e.NEUTRAL_MIGRATIONS = self.saved_mig
        self.e.MGATE = self.saved_gate

    def test_declaration_is_pinned_literal(self):
        """(a) 声明必须是**字面量**：它不能是运行时跟着 MGATE 漂的表达式。"""
        e = self.e
        v = e.LEGACY_GATE_EQUIV["math_gate"]
        self.assertIsInstance(v, str, "math_gate 必须是字符串字面量")
        # ★可失败性：声明里的值必须**曾经**等于某个真实 MGATE，不能是随手编的
        self.assertIn(v, [m["src"] for m in e.NEUTRAL_MIGRATIONS]
                      + [m["dst"] for m in e.NEUTRAL_MIGRATIONS] + [e.MGATE],
                      "声明/迁移里出现了从未在任何一版代码上出现过的指纹")

    def test_new_record_matches_math_gate(self):
        e = self.e
        self.assertTrue(e.gate_ok(dict(math_gate=e.MGATE)))
        self.assertFalse(e.gate_ok(dict(math_gate="000000000000")))

    def test_legacy_record_accepted(self):
        """(b) 登记在册的旧 gate 必须被接受（直连或经中性迁移）。"""
        e = self.e
        for g in e.LEGACY_GATE_EQUIV["gates"]:
            self.assertTrue(e.gate_ok(dict(gate=g)),
                            "旧记录 %s 在声明有效期内却不被接受" % g)

    def test_unknown_gate_rejected(self):
        e = self.e
        self.assertFalse(e.gate_ok(dict(gate="not-a-real-gate")))

    def test_negative_control_code_change_kills_legacy(self):
        """★负向对照(c)：真改了训练数值代码（MGATE 变）⇒ 旧记录必须立刻作废。

        ★这才是 P6 原本想测的东西，只是当年用恒真绑定测不出来。
        """
        e = self.e
        self.assertTrue(any(e.gate_ok(dict(gate=g))
                            for g in e.LEGACY_GATE_EQUIV["gates"]),
                        "前提失效：现在就接受不了旧记录，负向对照没意义")
        e.MGATE = "000000000000"                  # 假装训练数值代码又变了
        for g in e.LEGACY_GATE_EQUIV["gates"]:
            self.assertFalse(e.gate_ok(dict(gate=g)),
                             "MGATE 已变，旧记录 %s 却仍被接受 ⇒ 等价声明不会作废" % g)
        self.assertFalse(e.gate_ok(dict(math_gate=e.LEGACY_GATE_EQUIV["math_gate"])),
                         "MGATE 已变，带 math_gate 的旧记录却仍被接受")

    def test_negative_control_declared_change_kills_legacy(self):
        """★负向对照(d)：把声明指向另一个指纹 ⇒ 旧记录也必须作废。

        与 (c) 分开：(c) 动的是代码侧，(d) 动的是声明侧，两边都要测。
        """
        e = self.e
        e.LEGACY_GATE_EQUIV["math_gate"] = "000000000000"
        for g in e.LEGACY_GATE_EQUIV["gates"]:
            self.assertFalse(e.gate_ok(dict(gate=g)),
                             "声明已改指别的指纹，旧记录 %s 却仍被接受" % g)

    def test_migration_ends_at_current_gate(self):
        """(e) 每条中性迁移的 `dst` 必须钉在**当前** MGATE 上，否则它是漂的。"""
        e = self.e
        self.assertTrue(e.NEUTRAL_MIGRATIONS, "没有登记任何中性迁移")
        for mv in e.NEUTRAL_MIGRATIONS:
            self.assertEqual(mv["dst"], e.MGATE,
                             "迁移 %s→%s 的 dst 不等于当前 MGATE=%s ⇒ 该迁移应整体失效"
                             % (mv["src"], mv["dst"], e.MGATE))
            self.assertNotEqual(mv["src"], mv["dst"], "src==dst 的迁移是空操作")
            self.assertIn("evidence", mv, "迁移必须带证据")


if __name__ == "__main__":
    unittest.main()
