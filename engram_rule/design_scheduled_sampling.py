# -*- coding: utf-8 -*-
"""§69.42 scheduled sampling 的**静态设计** —— 先记账，不动一行代码。

── 为什么先写设计（AGENTS 规则 1）───────────────────────────────────
§69.41 实测出**强分布偏移**：教师强制第一步命中 83/200（41.5%），
其中只有 28 条（33.7%）在自由生成时也对 ⇒ 55 条翻脸。
诊断是 **exposure bias**：训练时每步都喂真值前缀，推理时喂自己的输出。

scheduled sampling 是标准对策：训练时以概率 p 用**模型自己的输出**替换真值前缀。
但**怎么替换**有几个选择，选错就没有效果甚至有害。本节把它们算清再动手。

── 三个必须定的设计参数 ────────────────────────────────────────────
| 参数 | 选项 | 本节的选择与理由 |
|---|---|---|
| **S1 在哪一步开始替换** | (a) 从第一步就替换 (b) 从答案段起点替换 (c) 随机步替换 | **(b)** —— 问题段必须是真值（否则模型连问题都读不准，另造一个偏差源）。只在 `[starts, len)` 内替换 |
| **S2 替换比例怎么定** | (a) 固定 p (b) 线性退火 p: 1→0 (c) 指数退火 | **(b)** —— 早期几乎全真值（学格式），后期几乎全自生成（贴近推理）。固定 p 早期会噪声太大 |
| **S3 梯度是否穿过替换点** | (a) 穿过（可微）(b) detach（不可微） | **(b)** —— 穿过会让梯度经「自己上一轮的错」回流，训练不稳（这是文献里已知的坑）；detach 只让**当前步**看到自己的输出 |

── 可证伪预测（先写死，失败即设计错）─────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **P1** | `p=0` 时与现有实现**逐位相同**（loss、末权重） | 不是纯加法 ⇒ 会改坏已有读数 |
| **P2** | 替换只发生在 `[starts, len)` 内，问题段一位不换 | 动了问题段 ⇒ 引入新偏差源 |
| **P3** | 同 seed 两跑逐位相同（确定性） | 不确定 ⇒ 读数不可信 |
| **P4** | `p=1`（全自生成）时 loss 明显**高于** `p=0` | 若更低 ⇒ 替换逻辑有错（不该比真值更容易） |
| **P5** | 退火到末端时，自由生成首字符命中率**上升** | 若不变 ⇒ exposure bias 不是首字符错的原因（§69.41 结论被推翻） |

── 代价（静态估算，不是实测）───────────────────────────────────────
自生成需要**每步一次前向**，而现在是整段一次前向。
答案段中位 ~40 token ⇒ 前向次数从 1 次涨到 ~40 次/样本。
`micro_batch=64` 时一步要 64×40 = 2560 次前向，**当前一步约 0.5 s ⇒ 预计 20~200 秒/步**。

★★**这是本设计最大的风险**：可能慢 40~400 倍。所以落地顺序必须是
  **先在 50 步的小规模上实测单步墙钟**，再决定是否值得全量跑。

用法：`python design_scheduled_sampling.py`（纯静态，只算数）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sched_sample_design.json")

# 出货/网格口径（来自落盘记录，不猜）
MICRO_BATCH = 64
BS = 256
ANSWER_TOK_MED = 40          # §69.38 M2 实测：生成长度中位 40
FWD_PER_MICRO = 4            # 现在：bs=256 / micro=64 ⇒ 4 个微批，每批 1 次前向
SEC_PER_STEP_NOW = 0.5       # d=128 / N=124,593 的实测 s/step 量级（REF 表）


def main():
    print("=" * 78)
    print("§69.42 scheduled sampling 静态设计（不动代码）")
    print("=" * 78)

    # ── 三个设计参数 ────────────────────────────────────────────────
    print()
    print("【设计参数】")
    print("  S1 替换范围：只在答案段 [starts, len) 内；问题段一位不换")
    print("     —— 理由：问题段必须是真值，否则另造一个偏差源")
    print("  S2 替换比例：线性退火 p: 1 → 0（随训练步数）")
    print("     —— 理由：早期几乎全真值（学格式），后期几乎全自生成（贴近推理）")
    print("  S3 梯度：**不穿过**替换点（detach）")
    print("     —— 理由：穿过会让梯度经「自己上一轮的错」回流，训练不稳")

    # ── 代价估算 ────────────────────────────────────────────────────
    print()
    print("【代价估算（静态，不是实测）】")
    fwd_now = FWD_PER_MICRO
    fwd_new = FWD_PER_MICRO * ANSWER_TOK_MED
    print("  现在：每步 %d 次前向（%d 个微批 × 1 次）" % (fwd_now, FWD_PER_MICRO))
    print("  改后：每步约 %d 次前向（每微批内逐步自生成 ~%d token）"
          % (fwd_new, ANSWER_TOK_MED))
    ratio = fwd_new / fwd_now
    est_lo = SEC_PER_STEP_NOW * ratio * 0.2      # 乐观：短序列/提前 EOS
    est_hi = SEC_PER_STEP_NOW * ratio            # 悲观：都跑满 40 token
    print("  单步墙钟：现在约 %.2f s ⇒ 预计 %.1f ~ %.1f s（%.0f~%.0f 倍）"
          % (SEC_PER_STEP_NOW, est_lo, est_hi, est_lo / SEC_PER_STEP_NOW,
             est_hi / SEC_PER_STEP_NOW))
    print("  ★★**这是本设计最大风险**：可能慢 %.0f~%.0f 倍。" % (est_lo / SEC_PER_STEP_NOW,
                                                        est_hi / SEC_PER_STEP_NOW))
    print("     ⇒ 落地顺序：**先跑 50 步小规模实测单步墙钟**，再决定是否全量。")

    # ── 全量训练时间 ────────────────────────────────────────────────
    print()
    print("【若全量跑的墙钟（按 N=124593 / d=128 / 19468 步）】")
    for label, sps in (("乐观 %.1f s/step" % est_lo, est_lo),
                       ("悲观 %.1f s/step" % est_hi, est_hi)):
        hrs = sps * 19468 / 3600.0
        print("  %-22s ⇒ %8.1f 小时（%.1f 天）" % (label, hrs, hrs / 24))
    print("  ★后台命令上限 3600 s ⇒ 每次只能跑约 %.0f 步，需分批"
          % (3600.0 / est_hi))

    facts = dict(
        design="scheduled sampling：只在答案段内、线性退火、梯度 detach",
        S1="替换范围 = [starts, len)，问题段不动",
        S2="替换比例 p 线性退火 1→0",
        S3="梯度不穿过替换点（detach）",
        cost=dict(micro_batch=MICRO_BATCH, bs=BS, answer_tok_med=ANSWER_TOK_MED,
                  fwd_now=fwd_now, fwd_new=fwd_new, ratio=ratio,
                  sec_per_step_now=SEC_PER_STEP_NOW,
                  est_sec_per_step=[round(est_lo, 1), round(est_hi, 1)],
                  hours_optimistic=round(est_lo * 19468 / 3600.0, 1),
                  hours_pessimistic=round(est_hi * 19468 / 3600.0, 1)),
        predictions=dict(
            P1="p=0 时与现有实现逐位相同（loss / 末权重）",
            P2="替换只在 [starts,len) 内，问题段一位不换",
            P3="同 seed 两跑逐位相同",
            P4="p=1 时 loss 明显高于 p=0",
            P5="退火末端自由生成首字符命中率上升"),
        evidence="§69.41：教师强制首字符命中 83/200(41.5%)，其中仅 28(33.7%)自由生成也对",
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 78)
    print("【结论】")
    print("=" * 78)
    print("  设计已定（S1/S2/S3），五条可证伪预测已登记。")
    print("  **尚未动一行代码。**")
    print()
    print("  但代价是硬风险：%.0f~%.0f 倍慢，全量 %.0f~%.0f 小时。"
          % (est_lo / SEC_PER_STEP_NOW, est_hi / SEC_PER_STEP_NOW,
             est_lo * 19468 / 3600.0, est_hi * 19468 / 3600.0))
    print("  在拿到「50 步小规模的实测单步墙钟」之前，这个改动**不该**全量跑。")
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
