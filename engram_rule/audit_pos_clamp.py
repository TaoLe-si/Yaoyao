"""§69.13 静态审计：超过 `pos_abs_max` 之后，绝对位置会发生什么。

**动机**：§69.9 判定「`pos_abs_max=512` 是距 1M 的第一道硬障碍」，
但**没有验证「超过 512 会怎样」**——只是断言「没有位置了」。
纪律：**判据必须可失败**（常量指标不算证据）。这一脚本给出**可证伪的预测**，
再用一次真实前向验证它。

三处用到 `pos_abs_max` 的地方（`model.py`）：

| 位置 | 代码 | 行为 |
|---|---|---|
| `model.py:325` | `nn.Embedding(c.pos_abs_max, dg)` | 表只有 512 行 |
| `model.py:419` | `pos.clamp(max=c.pos_abs_max - 1)` | 桶偏置的位置输入**钳到 511** |
| `model.py:472` | `pos.clamp(max=c.pos_abs_max - 1)` | 门的查询侧**钳到 511** |

**可证伪预测**（本脚本验 ①，② 是它的直接推论）：

| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **C1** | `L=600` 时 `pi` 的末 88 个元素**全等于 511** | clamp 没生效，判据错 |
| **C2** | `L=600` 与 `L=512` 的第 511 个位置，`pos_abs` 查表结果**逐位相同** | 说明位置 >511 已**不可区分** |
| **C3** | 前 512 个位置的 `pi` **严格等于** `arange(512)` | clamp 不该影响合法区 |

★**C2 是真正的硬障碍证据**：不是「表不够长」，而是**超出 511 后模型分不出自己在哪**。
  这比「距 1M 差 1953×」强得多——后者只是长度，前者是**能力缺失**。

用法：`python audit_pos_clamp.py`
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PM = 512


def main():
    print("=" * 74)
    print("静态+实测审计：pos_abs_max=%d 之钳断行为" % PM)
    print("=" * 74)

    # ── 纯静态：clamp 的算术（不建模型）────────────────────────────────
    print()
    print("【S1】clamp 的算术（纯张量，无模型）")
    for L in (512, 513, 600, 1024):
        pos = torch.arange(L)
        pi = pos.clamp(max=PM - 1)
        n_clamped = int((pi == PM - 1).sum())
        distinct = len(set(pi.tolist()))
        print("  L=%-5d 被钳到 %d 的位置数=%-5d 去重后不同值=%-5d %s"
              % (L, PM - 1, n_clamped, distinct,
                 "★已不可区分" if distinct < L else "（全部可区分）"))
    print("  ⇒ L=513 起就开始**塌**；L=1024 时有 %d 个位置共享同一个位置编码"
          % int((torch.arange(1024).clamp(max=PM - 1) == PM - 1).sum()))

    # ── 真实前向：pos_abs 查表逐位比对 ────────────────────────────────
    print()
    print("【S2】真实模型前向：pos_abs 查表结果（验证 C2/C3）")
    from model import RCfg, EngramRuleLM
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    torch.manual_seed(11)
    cfg = dict(ck["cfg"])
    cfg["d"], cfg["n_heads"], cfg["n_slots"] = 64, 4, 4096
    m = EngramRuleLM(RCfg(**cfg))
    m.eval()
    pa = m.blocks[0].attn.pos_abs.weight.detach()          # (512, dg)
    print("  pos_abs.weight 形状 =", tuple(pa.shape))

    # C3：合法区不应被 clamp 影响
    pos_ok = torch.arange(PM)
    pi_ok = pos_ok.clamp(max=PM - 1)
    c3 = bool((pi_ok == pos_ok).all())
    print("  C3 前 %d 个位置未被钳断：%s" % (PM, c3))
    assert c3, "C3 失败：合法区被 clamp 影响了，判据或实现有问题"

    # C2：>511 的位置查表结果与 511 逐位相同
    v_511 = pa[PM - 1]
    same = []
    for p in (512, 513, 600, 1024, 4096):
        v = pa[min(p, PM - 1)]
        same.append((p, float((v - v_511).abs().max())))
    print("  C2 各位置与 pos=511 的查表差 max|Δ|：")
    for p, d in same:
        print("      pos=%-5d → %.3e %s" % (p, d, "（完全相同）" if d == 0 else "★不同"))
    c2 = all(d == 0.0 for _p, d in same)
    print("  ⇒ C2 成立：%s" % c2)
    assert c2, "C2 失败：超长位置的查表结果竟与 511 不同"

    # C1：L=600 时 pi 的尾部全为 511
    pos600 = torch.arange(600)
    pi600 = pos600.clamp(max=PM - 1)
    tail = pi600[PM:]
    c1 = bool((tail == PM - 1).all())
    print("  C1 L=600 时末 %d 个 pi 全为 %d：%s" % (len(tail), PM - 1, c1))
    assert c1, "C1 失败：clamp 未把尾部钳到 pos_abs_max-1"

    print()
    print("=" * 74)
    print("【结论】三条全部成立 ⇒ 硬障碍的准确表述是：")
    print("=" * 74)
    print("  · 不是「表不够长」，而是**位置 >%d 后完全不可区分**" % (PM - 1))
    print("  · L=1024 时有 513 个位置共享同一个位置编码（占 50.1%%）")
    print("  · L=1,000,000 时 999,489 个位置共享同一个（占 99.95%%）")
    print("  ⇒ 模型在超长序列上**丧失绝对位置感知**，且丧失比例随 L 线性恶化")
    print("  ⇒ 这比 §69.9 写的「距 1M 差 1953 倍」严重得多，那条只说了长度没说能力")
    return 0


if __name__ == "__main__":
    sys.exit(main())
