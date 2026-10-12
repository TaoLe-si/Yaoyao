"""只读检查 `N=124593 × d=128` 这一格的状态，**不训练、不写盘、不留新文件**。

★为什么单独一个脚本：`exp_scaling_grid_v2.py` 一行 `python` 就开跑，而我现在
只是想确认「这一格会被跳过还是会跑」以及「恢复点还认不认」。误加 `--no-resume`
会把唯一的恢复点当不存在 ⇒ 从头重跑约 4758 s（刚才差点就这么做了）。

用途：续训前的一次静态确认。
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

CELL = dict(n=124593, d=128, wd=0.0, slots=4096, steps=19468, seq=512, bs=256,
            lr=0.003, seed=11, micro_batch=64, epochs=40.0)
RP = "resume_v2/resume_v2_n124593_d128_wd0p0_s19468_mb64_slots4096.pt"


def cell_done(rec, cell, slots):
    """逐字复刻 `exp_scaling_grid_v2.py:840-844` 的完成判定。

    ★为什么复刻而不 import：那个 `done()` 是 `main()` 里的**闭包**，
      依赖 `a.force` / `a.slots` 这些 argparse 命名空间，import 拿不到。
    ★复刻必须**逐字对齐**字段，否则这个探针会给出与真实调度不同的结论
      （那比不探还糟）。字段清单照抄源码那五行。
    """
    return (E2.gate_ok(rec) and rec.get("n") == cell["n"] and rec.get("d") == cell["d"]
            and rec.get("steps") == cell["steps"]
            and rec.get("weight_decay") == cell["wd"] and rec.get("n_slots") == slots)


def main():
    print("=" * 74)
    print("只读检查 N=124593 × d=128 这一格")
    print("=" * 74)
    print("  当前 math_gate = %s" % E2.MGATE)
    hits = []
    if os.path.exists(E2.OUT):
        with open(E2.OUT, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                if cell_done(j, CELL, CELL["slots"]):
                    hits.append(j)
    fin = bool(hits)
    print("  scaling_grid_v2.jsonl 里命中完成判定的记录 = %d 条" % len(hits))
    print("  ⇒ %s" % ("已完整，会被跳过" if fin else "**未完成，会跑这一格**"))
    assert not fin, "★前提变了：这一格已标记完成 ⇒ 不需要续训，请复核"

    if not os.path.exists(RP):
        print("  恢复点不存在 ⇒ 会从头跑满 %d 步" % CELL["steps"])
        return 0

    st = torch.load(RP, map_location="cpu", weights_only=False)
    key = st["key"]
    same = all(key.get(k) == v for k, v in CELL.items() if k in key)
    print("  恢复点存在: %s" % os.path.basename(RP))
    print("    step = %d / %d（已完成 %.2f%%）"
          % (st["step"], key["steps"], 100.0 * st["step"] / key["steps"]))
    print("    math_gate = %s ⇒ 与当前等价 = %s"
          % (st["math_gate"], E2._mgate_equivalent(st["math_gate"])))
    print("    恢复键与 CELL 一致 = %s" % same)
    print("    trace 已恢复点数 = %d" % len(st.get("trace") or []))
    assert same, "恢复键与当前 cell 不符 ⇒ load_resume 会拒绝，续不了"
    assert E2._mgate_equivalent(st["math_gate"]), "math_gate 不等价 ⇒ load_resume 会拒绝"
    print("  ⇒ 恢复点**可续**，剩余 %d 步（%.1f%%）"
          % (key["steps"] - st["step"], 100.0 * (key["steps"] - st["step"]) / key["steps"]))
    print()
    print("  ★这次是**断点续训**，不是重跑 ⇒ 不能加 --no-resume（会把恢复点当不存在）。")
    print("  ★墙钟上限约 3600 s，按同档 s/步 外推只能覆盖一部分；跑不完就再续一次，")
    print("    每 50 步落一次恢复点，进度不会丢。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
