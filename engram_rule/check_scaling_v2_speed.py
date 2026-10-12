# -*- coding: utf-8 -*-
"""§68 门禁：三处固定开销优化的**等价性**必须被证明，不是"看起来一样"。

改动都在 `exp_scaling_grid_v2.py`，**不碰出货路径**（`dialogue_small.py` 一行未改）。

| 编号 | 优化 | 等价性论证 | 负向对照（反例下必须翻转） |
|---|---|---|---|
| O1 | `build_engram` 只算一次，alive 跨模型复用 | 同一份 (x, lengths) ⇒ `slot_ids` 逐位相同 ⇒ `counts` 相同 ⇒ `alive = counts>=min_count` 相同 | 换一份 lengths（改一行）⇒ alive 必须不同 |
| O2 | d<=128 跳过 `probe_and_pick_micro` | `micro_batch_for(d)` 对 d<=128 **直接返回 64**，不依赖探测；探测只会返回同一个值 | d=256 时必须仍走探测（否则撞显存墙） |
| O3 | BPE 编码缓存 | 同一 (语料内容, tokenizer, max_len) ⇒ 编码逐位相同 | 改语料一行 ⇒ key 必须变 ⇒ 缓存必须未命中 |

用法：`python -m unittest check_scaling_v2_speed -v`
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if __import__("torch").cuda.is_available() else "cpu"


class AliveReuseTest(unittest.TestCase):
    """O1：alive 复用与重算必须逐位相同。"""

    def setUp(self):
        import exp_scaling_grid_v2 as e
        self.e = e

    def test_alive_reuse_bit_identical(self):
        e = self.e
        rows = e.load_rows()
        tr, _he = e.split_rows(rows)
        e.assert_no_leak(tr, _he)
        from tokenizers import Tokenizer
        import torch
        v11 = torch.load(os.path.join(HERE, "archive",
                                      "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                         map_location="cpu", weights_only=False)
        tok = Tokenizer.from_str(v11["tokenizer"]["json"])
        x, s, l = e.encode_cached(tr[:2000], tok, 512)

        e._TRAIN_X, e._TRAIN_S, e._TRAIN_L = x, s, l
        e._CK = v11
        m_ref = e.make_model(v11, 32)                     # 走 build_engram
        alive_ref = m_ref.engram.alive.clone()

        e._ALIVE_CACHE = alive_ref.clone()
        m_reuse = e.make_model(v11, 32, reuse_alive=True)  # 走缓存拷贝
        alive_reuse = m_reuse.engram.alive.clone()

        self.assertEqual(alive_ref.shape, alive_reuse.shape)
        self.assertTrue(bool((alive_ref == alive_reuse).all()),
                        "alive 复用与重算**不**逐位相同 ⇒ O1 不等价")
        self.assertGreater(int(alive_ref.sum()), 0, "alive 全 0，测了空集合")

    def test_alive_negative_control_must_differ(self):
        """★负向对照：把前 100 行的长度砍半 ⇒ alive 必须变。

        ★为什么不是只改最后一行（第一版那么写，**判据没翻转**）：
          500 行语料里最后一行（59 token）的 n-gram 槽**已被其余 499 行覆盖**，
          砍掉它的尾部不改变任何槽的存活状态 ⇒ alive 完全相同。
          这不是"判据无效"，是**扰动太弱**（规则 7：对照必须保证触发）。
          改成砍 100 行 ⇒ alive 从 4070 变 4068，判据翻转。
        """
        e = self.e
        import torch
        v11 = torch.load(os.path.join(HERE, "archive",
                                      "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                         map_location="cpu", weights_only=False)
        rows = e.load_rows()
        tr, _he = e.split_rows(rows)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_str(v11["tokenizer"]["json"])
        x, s, l = e.encode_cached(tr[:500], tok, 512)

        e._CK = v11
        e._TRAIN_X, e._TRAIN_S, e._TRAIN_L = x, s, l
        m1 = e.make_model(v11, 32)
        a1 = m1.engram.alive.clone()

        e._TRAIN_L = l.clone()
        e._TRAIN_L[:100] = (l[:100] // 2).clamp_min(2)   # 砍 100 行 ⇒ 少一大批 n-gram
        m2 = e.make_model(v11, 32)
        a2 = m2.engram.alive.clone()
        self.assertFalse(bool((a1 == a2).all()),
                         "砍了 100 行后 alive 竟完全不变 ⇒ 判据是常量，无效")
        self.assertNotEqual(int(a1.sum()), int(a2.sum()),
                            "alive 总数没变 ⇒ 扰动没真的改变 n-gram 集合")


class MicroBatchSkipTest(unittest.TestCase):
    """O2：d<=128 时跳过探测必须拿到同一个 mb。"""

    def test_skip_gives_same_micro_batch(self):
        import exp_scaling_grid_v2 as e
        for d in (32, 64, 128):
            self.assertEqual(e.micro_batch_for(d), 64,
                             "d=%d 的 mb 不是 64 ⇒ O2 的前提不成立" % d)
        self.assertEqual(e.micro_batch_for(256), 32,
                         "d=256 的 mb 不是 32 ⇒ 大档仍会撞墙")

    def test_d256_still_probes(self):
        """★负向对照：d>128 必须仍走探测分支（不能一律跳过）。"""
        import exp_scaling_grid_v2 as e
        # 静态检查源码里确实是 `if cell[\"d\"] <= 128` 而不是 `if True`
        src = open(os.path.join(HERE, "exp_scaling_grid_v2.py"),
                   encoding="utf-8").read()
        self.assertIn('if cell["d"] <= 128:', src,
                      "run_cell 里没有 d<=128 的条件分支 ⇒ O2 会误跳大档")
        self.assertNotIn("if True:", src, "出现了无条件分支")
        del src                              # 显式释放，避免 ResourceWarning


class EncodeCacheTest(unittest.TestCase):
    """O3：编码缓存命中时必须逐位相同，换语料必须失效。"""

    def setUp(self):
        import exp_scaling_grid_v2 as e
        self.e = e
        self.cache_path = e.ENCODED_CACHE
        self._existed = os.path.exists(self.cache_path)
        self._backup = None
        if self._existed:
            self._backup = self.cache_path + ".testbak"
            os.replace(self.cache_path, self._backup)

    def tearDown(self):
        if self._backup and os.path.exists(self._backup):
            os.replace(self._backup, self.cache_path)
        elif os.path.exists(self.cache_path) and not self._existed:
            os.remove(self.cache_path)

    def test_cache_hit_bit_identical(self):
        e = self.e
        rows = e.load_rows()[:300]
        from tokenizers import Tokenizer
        import torch
        v11 = torch.load(os.path.join(HERE, "archive",
                                      "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                         map_location="cpu", weights_only=False)
        tok = Tokenizer.from_str(v11["tokenizer"]["json"])
        a = e.encode_cached(rows, tok, 512)      # 未命中 ⇒ 算 + 落盘
        b = e.encode_cached(rows, tok, 512)      # 命中
        self.assertTrue(bool((a[0] == b[0]).all()), "编码缓存命中的 x 不一致")
        self.assertTrue(bool((a[2] == b[2]).all()), "编码缓存命中的 lengths 不一致")
        self.assertEqual(a[1].tolist(), b[1].tolist(), "编码缓存命中的 starts 不一致")

    def test_cache_key_changes_with_content(self):
        """★负向对照：改一行语料 ⇒ key 必须变 ⇒ 必须重新编码。"""
        e = self.e
        from tokenizers import Tokenizer
        import torch
        v11 = torch.load(os.path.join(HERE, "archive",
                                      "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                         map_location="cpu", weights_only=False)
        tok = Tokenizer.from_str(v11["tokenizer"]["json"])
        rows = e.load_rows()[:200]
        e.encode_cached(rows, tok, 512)
        cache1 = torch.load(self.cache_path, map_location="cpu", weights_only=False)

        rows2 = [dict(r) for r in rows]
        rows2[0]["question"] = rows2[0]["question"] + "（改一个字）"
        e.encode_cached(rows2, tok, 512)
        cache2 = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self.assertEqual(len(cache1), 1, "第一次应只有 1 个 key")
        self.assertEqual(len(cache2), 2,
                         "改了语料内容后 key 没有变 ⇒ 缓存会误命中，O3 不安全")
        self.assertNotEqual(set(cache1.keys()), set(cache2.keys()) - set(cache1.keys()),
                            "新旧 key 相同")


if __name__ == "__main__":
    unittest.main()
