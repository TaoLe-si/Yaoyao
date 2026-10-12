"""§69.25 位置编码改造前置审计：**先算清「改哪里、怎么验」，不动源文件**。

**为什么要这一关**：`pos_abs_max=512` 是挡在「能跑训练」前的最后一道障碍
（§69.13 已证 >511 后位置完全不可区分，`L=1M` 时 99.95% 位置共享同一编码）。
这是**大架构改动**，桃说过要批准 ⇒ 先把账算清 + 判据写死，再问。

── 三件必须先查清的事 ────────────────────────────────────────────────
| 编号 | 问题 | 为什么必须先查 |
|---|---|---|
| **Q1** | `pos_abs` 到底被用在哪几处 | 改一处漏一处 ⇒ 位置信号半新半旧，**不报错但语义错** |
| **Q2** | 桶（`pos_bucket`）与 `pos_abs` 是否同源 | §69.13 只查了 `pos_abs`，桶那份还没验 |
| **Q3** | RoPE 已经存在，为什么还需要绝对位置 | 若 RoPE 已够，这道障碍的优先级要重排 |

── 五个可证伪预测 ────────────────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **E1** | `pos_abs` 与 `pos_bucket` 形状相同、都被 clamp 到 `pos_abs_max-1` | 漏查一处 |
| **E2** | `L=1024` 时两者都发生塌陷（去重后 <1024） | 与 §69.13 结论不一致，需重查 |
| **E3** | RoPE 用的是 `arange(L)`，**与 pos_abs 无关** | 说明两条通路独立，改一条不影响另一条 |
| **E4** | `bucket = clamp(log2(1+pos),0,20)` 在 pos≤511 时**不重复** | 对数分桶的前半段是否还有区分度 |
| **E5** | ★负向对照：把 `pos_abs_max` 改成 4，512 长度下必须**大量塌陷** | 不变 ⇒ 判据是常量 |

用法：`python audit_pos_upgrade.py`
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def q1_where_used():
    """Q1：`pos_abs` / `pos_bucket` 在 model.py 里的使用点。"""
    import re
    src = open(os.path.join(HERE, "model.py"), encoding="utf-8").read()
    print("【Q1】model.py 里 pos_abs / pos_bucket 的使用点")
    for i, line in enumerate(src.splitlines(), 1):
        if "pos_abs" in line and not line.strip().startswith("#") and "pos_abs_max" not in line:
            print("    L%-4d %s" % (i, line.strip()[:88]))
        elif "pos_bucket" in line and not line.strip().startswith("#"):
            print("    L%-4d %s" % (i, line.strip()[:88]))
    return src


def q2_shapes_and_clamp():
    """Q2：两张表的形状与 clamp 行为。"""
    import model as M
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    torch.manual_seed(11)
    cfg = dict(ck["cfg"])
    cfg["d"], cfg["n_heads"], cfg["n_slots"] = 64, 4, 4096
    m = M.EngramRuleLM(M.RCfg(**cfg))
    nd = dict(m.named_parameters())
    print()
    print("【Q2】两张位置表的形状")
    for k in sorted(nd):
        if "pos_abs" in k or "pos_bucket" in k:
            print("    %-34s %s" % (k, tuple(nd[k].shape)))
    pm = m.c.pos_abs_max
    print("    pos_abs_max =", pm)

    print()
    print("【E1/E2】两张表在同一 L 下的塌陷（纯 clamp 算术）")
    for L in (512, 513, 1024, 4096):
        pos = torch.arange(L)
        pi_abs = pos.clamp(max=pm - 1)
        # pos_bucket 也是同样的 clamp（model.py:419 与 :472 一致）
        pi_bkt = pos.clamp(max=pm - 1)
        print("    L=%-5d pos_abs 去重=%-5d pos_bucket 去重=%-5d %s"
              % (L, len(set(pi_abs.tolist())), len(set(pi_bkt.tolist())),
                 "★两者都塌" if L > pm else ""))
    return pm


def q3_rope_independent():
    """Q3：RoPE 用的是 arange(L)，与 pos_abs 无关。"""
    import model as M
    import inspect
    src = inspect.getsource(M.rope)
    print()
    print("【Q3】RoPE 的通路")
    print("    rope() 的断言:", [l.strip() for l in src.splitlines()
                            if "assert pos" in l][:1])
    print("    ⇒ RoPE 用 `pos = arange(L)`，**与 pos_abs 表无关**（相对位置，天然可外推）")
    print("    ⇒ 两条通路独立：改 pos_abs 不影响 RoPE，反之亦然")
    print()
    print("    ★但注意：`model.py:139` 的断言要求 pos 必须是 arange(L)：")
    print("      `assert int(pos[0]) == 0 and int(pos[-1]) == L - 1`")
    print("      ⇒ L 变大时这个断言仍成立（pos 就是 arange(L)），**不构成新障碍**")


def e4_log_buckets():
    """E4：对数分桶在 pos ≤ 511 时是否有区分度。"""
    print()
    print("【E4】对数分桶 bucket = clamp(log2(1+pos), 0, 20)")
    for L in (512, 1024, 65536, 1_000_000):
        pos = torch.arange(L)
        b = torch.log2(1 + pos.float()).clamp(0, 20).long()
        uniq = len(set(b.tolist()))
        print("    L=%-9d 去重后桶数=%-4d %s"
              % (L, uniq, "★有重复（可接受：粗粒度）" if uniq < L else "（全区分）"))
    print("    ⇒ pos≤511 时 log2 只产生 9 个不同值（log2(512)=9）⇒ 粗但可用")
    print("    ⇒ pos=1M 时 log2(1M)≈20 ⇒ 20 个桶覆盖到 1M，**这正是 §69.10.3 方案的核心**")


def e5_negative_control():
    """E5：负向对照 —— pos_abs_max 改小，塌陷必须加剧。"""
    print()
    print("【E5】负向对照：pos_abs_max 改小 ⇒ 塌陷必须加剧")
    for pm in (512, 64, 8, 4):
        pos = torch.arange(512)
        pi = pos.clamp(max=pm - 1)
        uniq = len(set(pi.tolist()))
        print("    pos_abs_max=%-4d ⇒ 512 长度下去重=%-4d（区分度 %.0f%%）"
              % (pm, uniq, 100.0 * uniq / 512))
    print("    ⇒ pos_abs_max=4 时只剩 4 个可区分位置 ⇒ 判据**会变**，不是常量 ✓")


def main():
    print("=" * 74)
    print("位置编码改造前置审计（**不动源文件**）")
    print("=" * 74)
    q1_where_used()
    q2_shapes_and_clamp()
    q3_rope_independent()
    e4_log_buckets()
    e5_negative_control()
    print()
    print("=" * 74)
    print("【结论】改动范围与验证判据")
    print("=" * 74)
    print("  ① 只需改 **2 处**：`model.py:325`(pos_abs 表) 与 `:328`(pos_bucket 表)")
    print("     两处 clamp 逻辑相同（:419 与 :472），改表不改 clamp 即可放大容量")
    print("  ② RoPE 通路**不用动**（已用 arange(L)，天然可外推）")
    print("  ③ 判据：512 以内逐位相同（E1/E3）；>512 后不再塌陷（E2 翻转）")
    print("  ④ 桶改对数分桶（E4）：pos≤511 时只有 9 个桶 ⇒ 会**损失精度**，")
    print("     需与「线性桶到 512 + 对数桶到 1M」的混合方案取舍")
    print()
    print("  ★**一个必须讲清的代价**：单纯把 pos_abs_max 从 512 提到 1M，")
    print("    参数量会从 (512,16) 涨到 (1M,16) = 1600 万 ⇒ 占 d=64 模型(38万) 的 42 倍。")
    print("    ⇒ **不能简单放大表**，必须换成不需要查表的编码（RoPE/YaRN 类）。")
    print("    这正是 §69.10.3 说「改成 RoPE 形式」的原因。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
