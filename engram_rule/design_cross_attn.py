# -*- coding: utf-8 -*-
"""§69.44 跨注意力支路的**静态设计** —— 先记账，不动一行代码。

## 背景：为什么选这一条

§69.43 收口：六个训练侧方向全部排除（步数 / 正则 / 语料类型 / 信息可达性 /
scheduled sampling 代价 / 首字权重无目标），根因是**架构里没有一条「问题→答案」的通路**。
支撑读数：语义载体 : 规则侧 = **9.51 : 1**，语义全是 `emb`/`engram.val`/FFN 的副产品。

候选里 A（`use_qpool`）被 §69.40-D 同源配对否掉（可见性与质量相关性仅 6.7%），
所以选 **B：跨注意力** —— 让答案段的每个位置直接 attend 问题段。

## 设计（三个必须定的点）

| | 选择 | 理由 |
|---|---|---|
| **X1 插在哪** | `RuleBlock` 里、`attn` 之后、FFN 之前，作为**独立残差支路** | 不动现有 attn/Engram/窗口三条通路 ⇒ 关闭时可逐位还原 |
| **X2 K/V 从哪来** | **问题段**的 `h`（`[USER]` 之后、`[ASSISTANT]` 之前的那些位置） | 与 q_pool 同一套切段逻辑（`model.py:615-630` 已验证可用） |
| **X3 Q 从哪来** | **答案段**的 `h`（位置 >= `starts`） | 只让答案段提出问题；问题段自身不跨注意力（避免自相注意） |

**关键取舍**：X2/X3 意味着支路**只在答案段有输出**，问题段位置输出 0。
这与 `o = o * keep` 那类「整段乘掩码」不同 —— 它按段稀疏，**成本 O(L_q × L_a)**，
而 `L_q + L_a ≈ L` ⇒ 退化成 **O(L²)**。

★★**这是本设计最大的风险，必须先算**：出货是靠 O(L·W) 才在 L=16384 上比稠密快 1.86~1.93×
  （DESIGN §13/§43）。一旦答案段对问题段做全注意力，长序列下这条支路会**重新变成 O(L²)**，
  把已有的速度优势吃掉。

## 缓解手段（设计里就带上）

**X4 只对答案段的「前 K 个位置」做跨注意力**（`cross_k = 8`）。
理由：§69.39 实测崩在**第一个字**（LCP=0 占 64%），而后面的字是连锁偏。
⇒ 只需要让**开头几步**看得到问题，后面靠残差传递。
成本从 O(L_q × L_a) 降到 **O(L_q × K)**，K=8 时约为原来的 8/L_a。

## 可证伪预测（先写死，失败即设计错）

| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **C1** | `use_cross=False` 时与现有实现**逐位相同**（同权重同输入） | 不是纯加法 ⇒ 会改坏出货读数 |
| **C2** | 答案段前 K 个位置有非零梯度，问题段位置为 0 | 接错了段 |
| **C3** | 同 seed 两跑逐位相同 | 不确定 ⇒ 读数不可信 |
| **C4** | 中间张量 ≤ 现状的 2 倍（`cross_k=8` 必须真起限制作用） | 退化成了 O(L²) ⇒ 长序列会被拖死 |
| **C5** | held 抽样 200 的 char-F1 与 EM **上升** | 若不动 ⇒ 「缺问题→答案通路」也被否，记入否证清单 |

## 形状推导链（AGENTS §1A 步 D —— 上次就栽在这）

| 参数 | 形状 | 派生 |
|---|---|---|
| `cross_q` | `[d, d]` | `nn.Linear(d, d, bias=False)` |
| `cross_k` / `cross_v` | `[d, d]` | 同上 |
| `cross_o` | `[d, d]` | 同上 |
| 中间 `att` | `[B, K, L_q]` | K=`cross_k`=8，L_q = 问题段长度 |

新增 4 个 `[d,d]` 矩阵 ⇒ d=128 时 **4 × 16,384 = 65,536** 参数，
占现有 1,647,150 的 **3.98%** ⇒ state_dict 键变 ⇒ **旧 checkpoint 不可 resume**（已登记）。

## 显存（按 §69.29 的实测口径）

出货 `gate`+`up` = `(B=256, L=512, W=16, H=8, Dg=16)` = 1024 MB。
跨注意力的 `att` = `(256, 8, L_q)`，L_q 约 40 ⇒ 256×8×40×4 = **327 KB**。
⇒ 相对 1024 MB 可忽略。**C4 应该很容易过**，但那也意味着 X4 的 K=8 限制必须真的实现。

用法：`python design_cross_attn.py`（纯静态，只算数）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cross_attn_design.json")

D = 128
CROSS_K = 8
PARAMS_NOW = 1_647_150          # §65.2 实测（d=128 满步）
L_Q_MED = 40                    # §69.38 M2 实测：问题段约 40 token 量级


def main():
    print("=" * 78)
    print("§69.44 跨注意力支路静态设计（不动代码）")
    print("=" * 78)

    print()
    print("【设计】")
    print("  X1 位置：RuleBlock 内、attn 之后 FFN 之前，独立残差支路")
    print("  X2 K/V 来源：问题段 h（[USER] 后 / [ASSISTANT] 前）")
    print("  X3 Q 来源：答案段 h（位置 >= starts），只前 K=%d 个位置" % CROSS_K)
    print("  X4 限幅：cross_k=%d ⇒ 成本 O(L_q × K) 而非 O(L_q × L_a)" % CROSS_K)

    # ── 形状与参数（步 D）──────────────────────────────────────────
    print()
    print("【形状推导链】")
    print("  cross_q / cross_k / cross_v / cross_o 均为 nn.Linear(%d, %d, bias=False)"
          % (D, D))
    per = D * D
    new = 4 * per
    print("  每个 %s 参数，共 %s ⇒ 占现有 %s 的 %.2f%%"
          % ("{:,}".format(per), "{:,}".format(new),
             "{:,}".format(PARAMS_NOW), 100.0 * new / PARAMS_NOW))
    print("  ⇒ state_dict 键变 ⇒ 旧 checkpoint **不可 resume**（已登记）")

    # ── 显存（C4）──────────────────────────────────────────────────
    print()
    print("【显存（对照 §69.29 口径）】")
    B = 256
    gate_up = B * 512 * 16 * 8 * 16 * 4 / 2 ** 20
    att = B * CROSS_K * L_Q_MED * 4 / 2 ** 20
    print("  现有 gate+up = %.0f MB" % gate_up)
    print("  新增 att = (B=%d, K=%d, L_q=%d) = %.3f MB（%.4f%% of gate+up）"
          % (B, CROSS_K, L_Q_MED, att, 100 * att / gate_up))
    # ★C4 的负向对照：若 K 不设限（K=L_a），会变大多少
    L_a = 40
    att_full = B * L_a * L_Q_MED * 4 / 2 ** 20
    print("  ★负向对照：若 K 不设限（K=L_a=%d）⇒ att = %.1f MB（%.1f 倍）"
          % (L_a, att_full, att_full / att))
    assert att * 100 < gate_up, "C4 前提失效：K=8 竟没有压住显存"
    print("  ⇒ K=8 确实起限制作用 ⇒ C4 可过")

    # ── 预测 ────────────────────────────────────────────────────────
    facts = dict(
        design="跨注意力：答案段前 K 个位置 attend 问题段，独立残差支路",
        X1="RuleBlock 内 attn 之后 FFN 之前",
        X2="K/V 来自问题段 h",
        X3="Q 来自答案段前 K=%d 个位置" % CROSS_K,
        X4="cross_k=%d 限制成本 O(L_q×K)" % CROSS_K,
        shapes=dict(cross_q=[D, D], cross_k=[D, D], cross_v=[D, D], cross_o=[D, D],
                    att=[B, CROSS_K, L_Q_MED]),
        new_params=new, params_now=PARAMS_NOW, new_share=round(100.0 * new / PARAMS_NOW, 2),
        vram=dict(gate_up_mb=round(gate_up, 1), att_mb=round(att, 4),
                  att_full_if_no_k=round(att_full, 1)),
        predictions=dict(
            C1="use_cross=False 时与现有实现逐位相同",
            C2="答案段前 K 位有非零梯度，问题段位置为 0",
            C3="同 seed 两跑逐位相同",
            C4="中间张量 <= 现状 2 倍（K=8 真起限制）",
            C5="held 抽样 200 的 char-F1 与 EM 上升"),
        ckpt_incompatible=True,
    )
    with io_open(OUT) as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 78)
    print("【结论】")
    print("=" * 78)
    print("  设计已定，五条可证伪预测已登记。**尚未动一行代码。**")
    print("  代价：+%s 参数（%.2f%%）+ %.3f MB 显存 —— 都可忽略。" %
          ("{:,}".format(new), 100.0 * new / PARAMS_NOW, att))
    print("  风险：旧 checkpoint 不可 resume（键变），需重训。")
    print()
    print("  ★C5 是这条设计的**生死判据**：若 held F1 不动，")
    print("    「缺问题→答案通路」这条也否，连同前六次一起记入否证清单，不再投算力。")
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


def io_open(p):
    import io
    return io.open(p, "w", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
