# -*- coding: utf-8 -*-
"""★§69.28 门禁逻辑自身的判据 —— 包括一条**可失败性**的负向对照。

── 为什么要有这个 ────────────────────────────────────────────────────
`exp_scaling_grid_v2.py` 的 `LEGACY_GATE_EQUIV` 原来写的是 `math_gate=MGATE`，
而 `gate_ok` 判 `LEGACY_GATE_EQUIV["math_gate"] == MGATE` —— 两边是同一个值，
**恒真**。也就是说那道「训练数值代码一变、等价声明自动失效」的保险**从来没生效过**，
而注释把它写成生效了。这类假绿没有任何自动检查会抓到。

| 编号 | 判据 | 方向 |
|---|---|---|
| **G1** | 旧记录能经声明 + 中性迁移走到当前 MGATE | 必须接受 |
| **G2** | ★负向对照：MGATE 换成假的 ⇒ 迁移必须失效 | 必须拒绝 |
| **G3** | ★负向对照：复现旧写法的恒真绑定 | 恒真即证明旧写法无保护 |
| **G4** | 真实数据文件里的记录，接受/拒绝数符合预期 | 逐条点名 |
| **G5** | 未登记的 gate 一律拒绝 | 必须拒绝 |

用法：`python check_gate_binding.py`
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

MGATE = E2.MGATE
LEG = E2.LEGACY_GATE_EQUIV
MIG = E2.NEUTRAL_MIGRATIONS

# ── DATA 的两个数都是**快照**，会随新实验追加而过期 ──────────────────
# ★2026-10-13：scaling_grid_v2.jsonl 从 41 行长到 45 行（d=256 / d=512 扫描追加），
#   接受数 11 → 15。已逐条点名核实**新增的 4 条全部合法**：
#     · 7 条 gate 属 LEGACY_GATE_EQUIV['gates']（ff5eefd1604b ×6、59855015dfd8 ×1）
#     · 4 条 math_gate = 中性迁移的 src「4faabefdcd58」
#     · 4 条 math_gate = 当前 MGATE「eb7136d1f3f6」
#   被拒的 30 条 gate 全部未登记（10a7dbe4da97 ×18、4c8b6094fad8 ×6、87596e50af9c ×3、
#   92998206c5a0 ×2、9c07a7fc59be ×1）。
#   ⇒ **是检查过期，不是数据错**（AGENTS 规则 5 的又一次）。MGATE 全程未变。
# ★因此把**行数也纳入判据**：只钉接受数时，文件一长就看不出是「多了合法记录」
#   还是「放进了不该放的记录」；钉住两个数，漂移会直接暴露。
DATA = [("scaling_grid_v2.jsonl", 45, 15), ("capacity_decouple.jsonl", 4, 4)]


def main():
    print("=" * 78)
    print("门禁绑定审计（当前 MGATE = %s）" % MGATE)
    print("=" * 78)
    print("  LEGACY_GATE_EQUIV['math_gate'] = %r（类型 %s）"
          % (LEG["math_gate"], type(LEG["math_gate"]).__name__))
    for mv in MIG:
        print("  中性迁移: %s → %s" % (mv["src"], mv["dst"]))
        print("      why: %s" % mv["why"])

    # ── G1 ────────────────────────────────────────────────────────────
    print()
    print("【G1】旧记录（只有 gate）能否走通声明 + 迁移")
    rec = dict(gate=LEG["gates"][0])
    ok = E2.gate_ok(rec)
    print("  记录 gate=%s ⇒ gate_ok = %s" % (rec["gate"], ok))
    assert ok, "G1 失败：登记在册的旧 gate 被拒了"
    print("  ⇒ G1 成立")

    # ── G2：负向对照 —— 把 MGATE 换成假的，迁移必须整体失效 ────────────
    print()
    print("【G2】★负向对照：MGATE 变成别的值时，迁移必须失效")
    saved = E2.MGATE
    try:
        E2.MGATE = "deadbeefcafe"                 # 假装训练数值代码又变了
        leaked = [m["src"] for m in MIG if m["src"] == LEG["math_gate"]
                  and E2._mgate_equivalent(LEG["math_gate"])]
        print("  伪造 MGATE=deadbeefcafe ⇒ _mgate_equivalent('%s') = %s"
              % (LEG["math_gate"], E2._mgate_equivalent(LEG["math_gate"])))
        print("  ⇒ 仍然被接受的源指纹: %s（应为空）" % (leaked or "无"))
        assert E2._mgate_equivalent(LEG["math_gate"]) is False, \
            "G2 失败：MGATE 变了迁移却仍然放行 ⇒ 这道保险是恒真的"
        assert not E2.gate_ok(dict(gate=LEG["gates"][0])), \
            "G2 失败：MGATE 变了旧记录仍被接受"
    finally:
        E2.MGATE = saved
    print("  ⇒ 判据在反例下确实变 False ⇒ G2 成立（这是真保险，不是摆设）")

    # ── G3：复现旧写法的恒真绑定，证明 G2 不是空转 ──────────────────────
    print()
    print("【G3】★负向对照：旧写法（math_gate=MGATE）确实是恒真")
    old_style = dict(math_gate=MGATE)
    always_true = old_style["math_gate"] == MGATE
    print("  旧写法 dict(math_gate=MGATE)['math_gate'] == MGATE ⇒ %s（恒真）" % always_true)
    assert always_true is True, "G3 前提失效：本机复现不出恒真绑定"
    pinned_true = LEG["math_gate"] == MGATE
    print("  新写法 LEGACY_GATE_EQUIV['math_gate'] == MGATE ⇒ %s（**会**失效）" % pinned_true)
    assert isinstance(LEG["math_gate"], str) and LEG["math_gate"] != "MGATE"
    print("  ⇒ 旧写法无任何保护、新写法有 ⇒ G3 成立")

    # ── G4：真实数据文件逐条点名 ────────────────────────────────────────
    print()
    print("【G4】真实读数逐条判定")
    total_ok = total = 0
    for fn, expect_rows, expect in DATA:
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            print("  %-26s (不存在，跳过)" % fn)
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        ok = sum(1 for r in rows if E2.gate_ok(r))
        buckets = {}
        for r in rows:
            buckets.setdefault((r.get("gate"), r.get("math_gate")), [0, 0])
            buckets[(r.get("gate"), r.get("math_gate"))][0] += 1
            buckets[(r.get("gate"), r.get("math_gate"))][1] += int(E2.gate_ok(r))
        print("  %s: %d 条（预期 %d），接受 %d 条（预期 %d）"
              % (fn, len(rows), expect_rows, ok, expect))
        for (g, mg), (n, k) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
            print("      gate=%-14s math_gate=%-14s  %2d 条 → 接受 %2d" % (g, mg, n, k))
        assert len(rows) == expect_rows, (
            "G4 失败：%s 行数 %d ≠ 预期 %d ⇒ 要么有新实验追加（那就核对新增记录是否"
            "合法后更新快照），要么有记录被删" % (fn, len(rows), expect_rows))
        assert ok == expect, (
            "G4 失败：%s 接受数 %d ≠ 预期 %d。行数对得上而接受数不对 ⇒ "
            "**这才是真问题**：要么有该拒的被放行，要么有该接受的被拒。"
            % (fn, ok, expect))
        total_ok += ok
        total += len(rows)
    print("  ⇒ 合计 %d/%d 条被接受，G4 成立" % (total_ok, total))

    # ── G5：未登记的 gate 必须拒绝 ─────────────────────────────────────
    print()
    print("【G5】未登记的 gate 必须拒绝")
    bogus = [r for _, r in [("x", {"gate": "000000000000"})]]
    got = [E2.gate_ok(r) for r in bogus]
    print("  gate='000000000000' ⇒ %s（应为 False）" % got)
    assert got == [False], "G5 失败：未登记的 gate 被放行"
    print("  ⇒ G5 成立")
    print()
    print("=" * 78)
    print("【结论】门禁现在真的会失效：")
    print("  · LEGACY_GATE_EQUIV 钉**字面量**，不再自动跟随 MGATE（旧的恒真绑定已撤）")
    print("  · 本轮 think 改动走**中性迁移**，两端都是字面量，证据 = check_think_backcompat.py")
    print("  · %d/%d 条已有读数继续有效；改动 model.py 会立刻把它们全部作废" % (total_ok, total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
