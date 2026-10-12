"""§69.33 静态设计：与出货架构**同参数量**的标准 Transformer 对照臂。

── 为什么必须有这一条 ────────────────────────────────────────────────
桃 2026-10-11 连问三句「你确定架构没问题吗 / 你确定你不是猜的吗」。
我这几轮列的「问题清单」（门控、窗口非稀疏、无长程通路）**全部只有架构分析，
没有任何对照实验** ⇒ 「缺陷」这个词没有定义依据（缺陷 = 同任务上有别的做法更好）。

⇒ 唯一能回答的方式：造一个**同参数量、同数据、同 seed、同步数**的标准
  Transformer（MHA + softmax + RoPE + FFN + pre-LN），比 train 侧 probe NLL。

── 可证伪预测（先写死，再动手）─────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **A1** | 能在 ±5% 内配平到 1,647,150 参数 | 配不平 ⇒ 两臂不可比，对照无效 |
| **A2** | 同 seed 两跑逐位相同 | 不确定 ⇒ 读数不可信 |
| **A3** | 换 seed 必须改变结果（A2 的负向对照） | 不变 ⇒ A2 恒真，是假绿 |
| **A4** | 两臂用的是**同一份** x / starts / lengths | 不同 ⇒ 口径不一致，比较无效 |

── 硬约束 ────────────────────────────────────────────────────────────
· **不改任何出货文件**（model.py / dialogue_small.py / exp_scaling_grid_v2.py 一律不碰）。
  只新增本脚本 + `exp_baseline_ab.py` + `check_baseline_ab.py` + `baseline_ab.jsonl`。
· 标准 Transformer 的 softmax 注意力是 O(L^2)，`bs=256 / L=512` 下若物化完整矩阵
  需约 2.1 GB ⇒ 必须走 `F.scaled_dot_product_attention(is_causal=True)`，不物化。
· 判据只看 **train 侧**（桃的规矩：不再跑 eval）。

用法：`python probe_baseline_param_match.py`（纯算术，不加载模型）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "baseline_param_match.json")

VOCAB = 1024
TARGET = 1_647_150          # 出货可训练参数（probe_arch_figure.py 实测）
TOL = 0.05                  # ±5%


def count_params(d, L, ffn_mult, vocab=VOCAB):
    """标准 pre-norm Transformer 的可训练参数（无 bias 于线性层，与出货口径一致）。"""
    emb = vocab * d
    head = d * vocab
    per_layer = 4 * d * d + 2 * d * (ffn_mult * d)      # q,k,v,o + FFN 两层
    norms = 2 * L * 2 * d + 2 * d                       # 每层 2 个 LN + 末 LN
    return emb + head + L * per_layer + norms


def main():
    print("=" * 78)
    print("同参数量对照臂的配平表（目标 %s，容差 ±%.0f%%）"
          % ("{:,}".format(TARGET), 100 * TOL))
    print("=" * 78)
    cands = []
    for d in (128, 160, 192, 224, 256):
        for L in (2, 3, 4):
            for fm in (2, 3, 4):
                n = count_params(d, L, fm)
                dev = (n - TARGET) / float(TARGET)
                cands.append((abs(dev), dev, d, L, fm, n))

    cands.sort()
    print("  %6s %5s %9s %12s %9s %s"
          % ("d", "层数", "ffn_mult", "参数", "偏差", ""))
    for ad, dev, d, L, fm, n in cands[:8]:
        print("  %6d %5d %9d %12s %+8.2f%% %s"
              % (d, L, fm, "{:,}".format(n), 100 * dev, "OK" if ad <= TOL else "超容差"))

    best = cands[0]
    _, dev, d, L, fm, n = best
    print()
    print("  ⇒ 最优：d=%d / 层数=%d / ffn_mult=%d → %s 参数，偏差 %+.2f%%"
          % (d, L, fm, "{:,}".format(n), 100 * dev))

    # ★A1 的可失败性：必须真的落在容差内，且容差不是摆设
    assert abs(dev) <= TOL, \
        "A1 失败：最优配平偏差 %+.2f%% 超出 ±%.0f%% ⇒ 两臂参数不可比，对照无效" % (100 * dev, 100 * TOL)
    # ★负向对照：把目标改成 2 倍，最优解必须**不在**容差内（否则容差恒过）
    far = count_params(d, L, fm) * 2
    dev_far = (far - TARGET) / float(TARGET)
    assert abs(dev_far) > TOL, \
        "A1 负向对照失败：目标翻倍后偏差仍 %.2f%% ≤ %.0f%% ⇒ 容差恒真" % (100 * dev_far, 100 * TOL)
    print("  ⇒ A1 负向对照成立（目标翻倍后偏差 %+.2f%%，判据确实会翻）" % (100 * dev_far))

    # 出货架构的对照口径
    ship = dict(d=128, n_layers=2, n_slots=4096, ngram_max=2, n_hash=2,
                pos_abs_max=512, window_max=16, n_buckets=16, bucket_step=32,
                params=TARGET)
    print()
    print("  出货架构（probe_arch_figure.py 实测）：")
    print("    d=%d  层数=%d  n_slots=%d  可训练参数 = %s"
          % (ship["d"], ship["n_layers"], ship["n_slots"], "{:,}".format(ship["params"])))
    print("    · engram 表占 63.66%（1,048,576）—— 对照臂没有这部分，所以它的 d 必须更大")
    print("    · 对照臂把同样的参数预算花在 attention + FFN 上（稠密）")

    facts = dict(
        target=TARGET, tol=TOL,
        best=dict(d=d, layers=L, ffn_mult=fm, params=n, dev=dev),
        top8=[dict(d=a[2], layers=a[3], ffn_mult=a[4], params=a[5], dev=a[1]) for a in cands[:8]],
        ship=ship,
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
