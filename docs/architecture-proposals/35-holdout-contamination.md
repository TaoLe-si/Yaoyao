# 35 — 留出探测污染：一次必须纠正的测量错误

生成时间 2026-09-12。本文记录一个**由我自己引入、并已用哈希证实的测量错误**，以及它的影响范围与纠正后的口径。

---

## 一、发现

我一直把三个文件当作"留出探测"来测遗忘曲线（`scripts/forget_sweep.ps1`、
`scripts/paired_forget_test.mjs`、以及各阶段的 nll_probe 调用）：

| 角色 | 文件 |
|---|---|
| wiki 探测 | `data/stage_wiki/shard_00000.bin` |
| 推理探测 | `data/stage2_reason_src/shard_00000.bin` |
| alpaca 探测 | `build/alp_big/shard_00000.bin` |

**MD5 实测：这三个文件就是训练课程的分片本身。**

| MD5（前16位） | 文件 A | 文件 B |
|---|---|---|
| `9D925DCCD101397D` | `data/stage_wiki/shard_00000.bin` | `data/stage2_reason/shard_00000.bin` |
| `9113B4A37AC62007` | `data/stage2_reason_src/shard_00000.bin` | `data/stage2_reason/shard_00013.bin` |
| `F28CE0847159069B` | `build/alp_big/shard_00000.bin` | `data/stage2_reason/shard_00017.bin` |
| `BFEDD4F9CDC0A5A7` | `build/alp_big/shard_00001.bin` | `data/stage2_reason/shard_00018.bin` |

即：`data/stage2_reason` 这个训练课程，是把三个"探测文件"按 13/4/8 的比例拼起来的
（0–12 是 wiki、13–16 是推理、17–24 是 alpaca 的两个文件各重复 4 次）。
尺寸也逐一吻合（如 stage_wiki/0 与 stage2_reason/0 同为 42,816,226 字节）。

## 二、更彻底的结论：该模型不存在干净的域内留出集

从各次运行日志的 `SHARD_TRAIN_START` 统计训练分片并集：

- `build/s2_night1` 有**三次** `SHARD_TRAIN_START`（`shards=15 steps=1918`、`shards=21 steps=1000`、`shards=21 …`），
  其实际训练分片并集为 **0–24 全部 25 片**。
- 因此 wiki(0–12)、推理(13–16)、alpaca(17–24) **全部参与过训练**。

**结论：训练语料内不存在任何一篇从未被训练过的文档。** 该模型的任何域内 NLL 都是训练集 NLL。

## 三、影响范围

**被推翻（不得再作为泛化证据引用）**：
- 阶段2 三阶段留出 NLL 曲线（①②③ 各段的 wiki/推理/alpaca 数值）
- "维基段已收敛（−1.99%）"
- "推理在 alpaca 段 +159.7%"
- "alpaca 段 −23.0%"
- `build/forget_curve.tsv`（98 点）与 `build/forget_curve_t1.tsv` 的全部绝对值
- 一切以"留出"称呼的 wiki 60/40 篇 NLL（6.748 / 4.818 / 4.738 / 5.243 / 5.586）

**不受影响（仍是有效证据）**：
- **alpaca 重复训练的 epoch-over-epoch 下降**（4.673→4.376→4.318→4.292）—— 那是**训练 NLL**，
  本来就是训练量证据，与"留出"无关。"欠训练在指令域成立"的结论保留。
- 各阶段的**训练** NLL 逐片均值（wiki 4.66–4.75 / 推理 1.04–1.37 / alpaca 4.29–4.67）。
- 所有以 `eval_real.mjs`（`data/alpaca_heldout.jsonl`）、`eval_math.mjs`
  （Ape210K 官方 test）为基础的质量指标 —— 那些数据与训练确实不相交。

## 四、结论其实被强化，而不是削弱

关键点：**这些曲线是在训练数据上测的，而后期阶段它们的 NLL 仍然上升**
（wiki 4.73 → 5.42，+14.6%；推理 0.96 → 2.49）。

模型在**自己已经学过的数据**上变差了 —— 这比"留出集上变差"**更强**地证明发生了干扰/遗忘：
它连旧知识的拟合能力都被后续训练破坏了。所以：
- **定性结论（存在严重干扰，且推理域最严重）成立且被加强。**
- **定量结论（幅度百分比）作废**，因为训练集 NLL 不是泛化指标，其"上升"幅度不能外推为泛化损失幅度。

## 五、纠正后的质量验证口径

后续一切"模型质量"必须以**与训练确实不相交**的数据为准：

| 域 | 验证数据 | 是否不相交 | 依据 |
|---|---|---|---|
| 指令 | `data/alpaca_heldout.jsonl` (n=400) | 基本是（约 1.3% 泄漏，已记录） | 取自 alpaca_zh.json 末尾 |
| 推理 | `E:/taovm-data/real_reason/ape210k_test_probe.jsonl` | **是** | 训练用 ape210k **train** split |
| 推理 | `E:/taovm-data/real_reason/gsm8k_test.jsonl` | **是** | 训练用 gsm8k **train** split |
| 维基 | **无** | — | 全部 254,107 篇均已训练 |

维基域若要诚实度量，必须在**下一次造语料时**用 `build_corpus --seal` 预留留出集；
`build_corpus` 本身已提供 `--seal FILE`（"封存实体清单，评测 held-out 用"），
此前从未被使用过。

> 附注：`wiki_docs.txt` 有 254,540 篇而只打包了 254,107 篇，差的 433 篇是
> `build_corpus` 按 `--min-bytes 64` / 近似去重 / 退化重复上限**过滤掉**的短文档，
> 不是随机留出，用它当留出会有分布偏置，故不采用。

## 六、收敛门控的留出集同样受污染（已记录）

本轮 `TAO_HOLDOUT=128` 取每片**末 128 篇**排除出训练。但 `s2_night1` 上一轮已把
全部分片（含这 128 篇）训过，所以这批"留出"是**模型见过的数据**。

处置：
- 该留出仍能检出**本轮新发生的**过拟合（这 128 篇本轮未参与更新），
  但信号偏乐观 —— 判据阈值用**配对差的标准误**，故不会因基准低而失准；
- 它在 400 步处正确判出 `learning`（gain 0.0735 > need 0.0260），机制可用；
- 真正的干净留出须待下一次造语料时用 `--seal` 预留。
