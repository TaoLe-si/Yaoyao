"""§66 的零 GPU 检查：先验证判据会在反例下翻转，再读历史 15 格。"""
import json
import math
import os
import unittest

import probe_scaling_fit as subject

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = dict(steps=200, seed=11, bs=256, n_probe=82, probe_tokens=5200,
              lr=0.003, head_dim=16, micro_batch=64)
NS = (50, 100)
DS = (32, 64, 128)


def rows(values):
    """手算的 2x3 完整网格；每格配方同值，仅 N/d/NLL 不同。"""
    return [dict(gate="fixture", n=n, d=d, probe_nll=y, **RECIPE)
            for n, triplet in values
            for d, y in zip(DS, triplet)]


class ScalingReauditTest(unittest.TestCase):
    def audit(self, data):
        self.assertTrue(callable(getattr(subject, "audit_grid", None)),
                        "旧分析器缺少可失败的 audit_grid 判据")
        return subject.audit_grid(data, gate="fixture", ns=NS, ds=DS, recipe=RECIPE)

    def test_positive_control_strictly_decreasing_grid_passes(self):
        data = rows(((50, (3.0, 2.0, 1.0)),
                     (100, (2.5, 1.5, 0.5))))
        result = self.audit(data)
        self.assertTrue(result["capacity_nonincreasing"])
        self.assertTrue(result["data_nonincreasing"])
        self.assertEqual(result["capacity_end_deltas"], {50: -2.0, 100: -2.0})
        self.assertEqual(result["capacity_adjacent_deltas"][50], [-1.0, -1.0])
        self.assertEqual(result["data_adjacent_deltas"][32], [-0.5])

    def test_reverse_d_labels_flips_capacity_verdict(self):
        decreasing = rows(((50, (3.0, 2.0, 1.0)),
                           (100, (2.5, 1.5, 0.5))))
        reversed_rows = [dict(r, d=DS[::-1][DS.index(r["d"])]) for r in decreasing]
        result = self.audit(reversed_rows)
        self.assertFalse(result["capacity_nonincreasing"])
        self.assertEqual(result["capacity_end_deltas"], {50: 2.0, 100: 2.0})
        self.assertTrue(result["data_nonincreasing"])

    def test_reverse_n_labels_flips_data_verdict(self):
        decreasing = rows(((50, (3.0, 2.0, 1.0)),
                           (100, (2.5, 1.5, 0.5))))
        reversed_rows = [dict(r, n=150 - r["n"]) for r in decreasing]
        result = self.audit(reversed_rows)
        self.assertFalse(result["data_nonincreasing"])
        self.assertEqual(result["data_adjacent_deltas"][32], [0.5])
        self.assertTrue(result["capacity_nonincreasing"])

    def test_reject_stale_gate_duplicate_and_changed_probe_tokens(self):
        good = rows(((50, (3.0, 2.0, 1.0)),
                     (100, (2.5, 1.5, 0.5))))
        for bad in (good[:1] + [dict(good[1], gate="stale")] + good[2:],
                    good + [dict(good[0])],
                    good[:1] + [dict(good[1], probe_tokens=5199)] + good[2:]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.audit(bad)

    def test_reject_incomplete_grid_wrong_recipe_and_nonfinite_nll(self):
        good = rows(((50, (3.0, 2.0, 1.0)),
                     (100, (2.5, 1.5, 0.5))))
        for bad in (good[:-1],
                    good[:1] + [dict(good[1], steps=400)] + good[2:],
                    good[:1] + [dict(good[1], probe_nll=math.nan)] + good[2:]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.audit(bad)

    def test_historical_grid_anchor_has_correct_sign_and_no_new_eval(self):
        self.assertTrue(callable(getattr(subject, "audit_grid", None)),
                        "旧分析器缺少可失败的 audit_grid 判据")
        with open(os.path.join(HERE, "scaling_grid_v1.jsonl"), encoding="utf-8") as f:
            historical = [json.loads(line) for line in f if line.strip()]
        result = subject.audit_grid(historical, gate="5ebe895bde34",
                                    ns=(50, 100, 200, 400, 738), ds=DS,
                                    recipe=RECIPE)
        self.assertEqual(result["grid_cells"], 15)
        self.assertFalse(result["capacity_nonincreasing"])
        self.assertAlmostEqual(result["capacity_end_deltas"][738],
                               0.9908757209777832, delta=5e-7)
        self.assertTrue(all(v > 0 for v in result["capacity_end_deltas"].values()))


if __name__ == "__main__":
    unittest.main()
