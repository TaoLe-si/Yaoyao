"""★ 验证 gate 机制是否真能挡「改代码后跑数据」—— 负向对照(规则 7)。

**为什么必须有这个脚本**: gate 只是测量脚本里算出来的一串 hex。我把它写进 docstring
说"改代码后旧数据自动作废", 但**从未验证过**。一个永远没人验过的安全机制不是机制。

**判定方法(可失败, 不靠眼看)**:
  1. 用当前 gate 读落盘记录 —— 必须全部有效(正对照)
  2. 把 `exp_scaling_grid.py` 复制到临时文件, 往里面 append 一行注释,
     算出**被改过**的 gate —— 必须与当前 gate 不同
  3. 用被改过的 gate 去读同一份 JSONL —— 必须全部作废(负向对照)
  4. 还原: 删掉临时文件, 确认原文件 sha256 未变

**三条判据全部 PASS 才算 gate 有效**:
  G1 正对照: 当前 gate 下 15/15 条有效
  G2 扰动有效: 追加 1 行注释后 gate 必须变
  G3 负向对照: 用改过的 gate 读, 15/15 条必须作废
  G4 可还原: 原文件 sha256 不变(脚本没留下副作用)
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = os.path.join(HERE, "scaling_grid_v1.jsonl")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def gate_of(paths):
    """与 exp_scaling_grid._gate 同算法, 但作用于**指定的路径列表**。"""
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:12]


def read_with(gate):
    ok = stale = 0
    with open(SRC, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if j.get("gate") == gate:
                ok += 1
            else:
                stale += 1
    return ok, stale


def main():
    import exp_scaling_grid as exp

    print("=" * 76)
    print("gate 机制的负向对照（规则 7：判据必须可失败）")
    print("=" * 76)

    results = {}

    # ---- G1 正对照: 当前 gate 下 15/15 有效 ----
    ok, stale = read_with(exp.GATE)
    results["G1"] = (ok == 15 and stale == 0)
    print("G1 正对照   当前 gate=%s 读落盘: 有效 %d / 作废 %d   [%s]"
          % (exp.GATE, ok, stale, "PASS" if results["G1"] else "FAIL"))

    # ---- G2 + G3 扰动 gate 必须变、旧数据必须全作废 ----
    tmpdir = tempfile.mkdtemp(prefix="gatecheck_")
    try:
        copy = os.path.join(tmpdir, "exp_scaling_grid.py")
        shutil.copyfile(os.path.join(HERE, "exp_scaling_grid.py"), copy)
        with open(copy, "a", encoding="utf-8") as f:
            f.write("\n# 负向对照: 只加一行注释, 模拟「改代码后没重跑检查」\n")
        alt = gate_of([copy, os.path.join(HERE, "model.py"),
                       os.path.join(HERE, "dialogue_small.py")])

        results["G2"] = (alt != exp.GATE)
        print("G2 扰动有效 追加 1 行注释后 gate=%s  与当前 %s   %s   [%s]"
              % (alt, exp.GATE, "不同" if alt != exp.GATE else "★相同 ⇒ gate 无效",
                 "PASS" if results["G2"] else "FAIL"))

        ok2, stale2 = read_with(alt)
        results["G3"] = (ok2 == 0 and stale2 == 15)
        print("G3 负向对照 用改过的 gate 读同一份 JSONL: 有效 %d / 作废 %d   [%s]"
              % (ok2, stale2, "PASS" if results["G3"] else "FAIL"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- G4 可还原: 原文件没被动过 ----
    before = sha(os.path.join(HERE, "exp_scaling_grid.py"))
    results["G4"] = (gate_of([os.path.join(HERE, "exp_scaling_grid.py"),
                              os.path.join(HERE, "model.py"),
                              os.path.join(HERE, "dialogue_small.py")]) == exp.GATE
                     and before == sha(os.path.join(HERE, "exp_scaling_grid.py")))
    print("G4 可还原   exp_scaling_grid.py sha256 未变、gate 仍为 %s   [%s]"
          % (exp.GATE, "PASS" if results["G4"] else "FAIL"))

    print()
    n_pass = sum(1 for v in results.values() if v)
    print("⇒ %d / %d 项 %s" % (n_pass, len(results),
                                "PASS：gate 机制确实能挡「改代码后跑数据」"
                                if n_pass == len(results)
                                else "FAIL：gate 机制不可靠"))
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())