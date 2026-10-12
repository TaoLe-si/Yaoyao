# 架构清点：标量 / 向量 / 矩阵 / 在激活（2026-10-10）

脚本 `engram_rule/probe_arch_audit.py`，**零训练**，秒级。
出货配置来自 `engram_rule_dialogue_bpe_answer_disjoint_v11.pt`。

> ## ⚠ 本文的口径与适用范围（2026-10-10 §65 补注，**先读这段再看数字**）
>
> 1. **本文清点的是 v11 出厂配置 `d=64`**（n_heads=4），**不是当前基准**。
>    当前基准是 `engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st400.pt`
>    （d=128 / n_heads=8 / n_layers=2 / seed=11 / 400 步）。
>    两者的参数数、窗口读数**都不可互相套用**。
> 2. **「标量 / 向量 / 矩阵」的分类口径与 §65 不同。**
>    本文把形状 `(1,)` 的 `content_gate.bias` 记为**标量**（故标量 6 个）；
>    §65 的 `probe_arch_figure.py` 按**张量维数 `dim()`** 分类，
>    `dim()==0` 才算标量、`dim()==1` 一律算向量（故 d=128 上标量 4 个）。
>    **两个口径都对，只是尺子不同** —— 引用时必须连口径一起引。
> 3. 本文第 5 节实测 `start_bucket` 取值 = `{0}`（**d=64 / v11**）。
>    §65 在 d=128 / 400 步基准上实测为 **`{0, 1}`**。
>    这不是矛盾：**桶选择是学出来的，d 与训练步数都变了，取值集合就会变**。
>    ⇒ 以 §65 的 `{0, 1}` 为当前基准读数，本文的 `{0}` 只对 v11 成立。
> 4. 当前基准（d=128）的完整清点与架构图见 **`DESIGN.md` §65**，
>    原始数据见 `arch_figure_facts.json`，图见 `arch_figure_engram_rule.png`。

## 1. 配置快照

| 键 | 值 |
|---|---:|
| `vocab` | 1024 |
| `d` | 64 |
| `n_layers` | 2 |
| `n_heads` | 4 |
| `ffn_mult` | 2 |
| `rope_dim` | 8 |
| `ngram_max` | 2 |
| `n_hash` | 2 |
| `n_slots` | 4096 |
| `n_buckets` | 16 |
| `gate_rank` | 16 |
| `pos_abs_max` | 512 |
| `window_max` | 16 |
| `bucket_step` | 32 |
| `near_first` | 2.0 |
| `rule_scale_init` | 2.0 |
| `lam_init` | 1.0 |
| `use_pos_abs` | True |
| `use_pos_bucket` | True |
| `clamp_window` | True |
| `local_norm` | True |
| `use_qpool` | — |

## 2. 按种类汇总

| 种类 | 参数个数 | 元素总数 | 占比 |
|---|---:|---:|---:|
| **标量** | 6 | 6 | 0.00% |
| **向量** | 20 | 1,192 | 0.16% |
| **矩阵** | 34 | 766,080 | 99.84% |
| 合计 | 60 | 767,278 | 100% |

★**参与前向的元素只占 100.0%**（判据：对两个不同输入求梯度，两次都非零才算激活）。

## 3. 矩阵参数（按元素数降序）

| 参数名 | shape | 元素数 | 激活 |
|---|---|---:|---|
| `engram.key.weight` | 4096×64 | 262,144 | ✓ |
| `engram.val.weight` | 4096×64 | 262,144 | ✓ |
| `emb.weight` | 1024×64 | 65,536 | ✓ |
| `head.weight` | 1024×64 | 65,536 | ✓ |
| `blocks.0.attn.pos_abs.weight` | 512×16 | 8,192 | ✓ |
| `blocks.0.attn.pos_bucket.weight` | 512×16 | 8,192 | ✓ |
| `blocks.0.ffn.0.weight` | 128×64 | 8,192 | ✓ |
| `blocks.0.ffn.2.weight` | 64×128 | 8,192 | ✓ |
| `blocks.1.attn.pos_abs.weight` | 512×16 | 8,192 | ✓ |
| `blocks.1.attn.pos_bucket.weight` | 512×16 | 8,192 | ✓ |
| `blocks.1.ffn.0.weight` | 128×64 | 8,192 | ✓ |
| `blocks.1.ffn.2.weight` | 64×128 | 8,192 | ✓ |
| `blocks.0.attn.q.weight` | 64×64 | 4,096 | ✓ |
| `blocks.0.attn.k.weight` | 64×64 | 4,096 | ✓ |
| `blocks.0.attn.v.weight` | 64×64 | 4,096 | ✓ |
| `blocks.0.attn.o.weight` | 64×64 | 4,096 | ✓ |
| `blocks.0.attn.direct.weight` | 64×64 | 4,096 | ✓ |
| `blocks.1.attn.q.weight` | 64×64 | 4,096 | ✓ |
| `blocks.1.attn.k.weight` | 64×64 | 4,096 | ✓ |
| `blocks.1.attn.v.weight` | 64×64 | 4,096 | ✓ |
| `blocks.1.attn.o.weight` | 64×64 | 4,096 | ✓ |
| `blocks.1.attn.direct.weight` | 64×64 | 4,096 | ✓ |
| `blocks.0.attn.bucket_proj.weight` | 16×64 | 1,024 | ✓ |
| `blocks.1.attn.bucket_proj.weight` | 16×64 | 1,024 | ✓ |
| `blocks.0.attn.wg.weight` | 16×16 | 256 | ✓ |
| `blocks.0.attn.wq.weight` | 16×16 | 256 | ✓ |
| `blocks.0.attn.wu.weight` | 16×16 | 256 | ✓ |
| `blocks.0.attn.wd.weight` | 16×16 | 256 | ✓ |
| `blocks.1.attn.wg.weight` | 16×16 | 256 | ✓ |
| `blocks.1.attn.wq.weight` | 16×16 | 256 | ✓ |
| `blocks.1.attn.wu.weight` | 16×16 | 256 | ✓ |
| `blocks.1.attn.wd.weight` | 16×16 | 256 | ✓ |
| `blocks.0.attn.content_gate.weight` | 1×64 | 64 | ✓ |
| `blocks.1.attn.content_gate.weight` | 1×64 | 64 | ✓ |

## 4. 标量与向量

| 参数名 | shape | 元素数 | 种类 | 激活 |
|---|---|---:|---|---|
| `blocks.0.attn.rule_scale` | () | 1 | 标量 | ✓ |
| `blocks.0.attn.lam` | () | 1 | 标量 | ✓ |
| `blocks.0.attn.content_gate.bias` | 1 | 1 | 标量 | ✓ |
| `blocks.1.attn.rule_scale` | () | 1 | 标量 | ✓ |
| `blocks.1.attn.lam` | () | 1 | 标量 | ✓ |
| `blocks.1.attn.content_gate.bias` | 1 | 1 | 标量 | ✓ |
| `blocks.0.attn.head_w` | 4 | 4 | 向量 | ✓ |
| `blocks.1.attn.head_w` | 4 | 4 | 向量 | ✓ |
| `blocks.0.attn.bucket_proj.bias` | 16 | 16 | 向量 | ✓ |
| `blocks.1.attn.bucket_proj.bias` | 16 | 16 | 向量 | ✓ |
| `blocks.0.norm1.weight` | 64 | 64 | 向量 | ✓ |
| `blocks.0.norm1.bias` | 64 | 64 | 向量 | ✓ |
| `blocks.0.attn.gate` | 64 | 64 | 向量 | ✓ |
| `blocks.0.norm2.weight` | 64 | 64 | 向量 | ✓ |
| `blocks.0.norm2.bias` | 64 | 64 | 向量 | ✓ |
| `blocks.0.ffn.2.bias` | 64 | 64 | 向量 | ✓ |
| `blocks.1.norm1.weight` | 64 | 64 | 向量 | ✓ |
| `blocks.1.norm1.bias` | 64 | 64 | 向量 | ✓ |
| `blocks.1.attn.gate` | 64 | 64 | 向量 | ✓ |
| `blocks.1.norm2.weight` | 64 | 64 | 向量 | ✓ |
| `blocks.1.norm2.bias` | 64 | 64 | 向量 | ✓ |
| `blocks.1.ffn.2.bias` | 64 | 64 | 向量 | ✓ |
| `norm.weight` | 64 | 64 | 向量 | ✓ |
| `norm.bias` | 64 | 64 | 向量 | ✓ |
| `blocks.0.ffn.0.bias` | 128 | 128 | 向量 | ✓ |
| `blocks.1.ffn.0.bias` | 128 | 128 | 向量 | ✓ |

## 5. 通路开关

| 开关/通路 | 取值 | 含义 |
|---|---:|---|
| `use_pos_abs` | True | 门的查询侧加绝对位置 |
| `use_pos_bucket` | True | 位置进桶偏置（决定窗口起点） |
| `clamp_window` | True | 因果夹紧（C28） |
| `local_norm` | True | 窗口内逐通道归一化（C29） |
| `use_qpool` | None | ★问题池化通路（§61）——**出货模型没开** |

## 6. 实测激活状态

| 量 | 读数 |
|---|---:|
| Engram 存活槽 | 4095 / 4096（100.0%） |
| `start_bucket` 取值分布 | {0: 65536} |
| 本批规则路径是否在跑 | True |

## 7. 诚实边界

- 「激活」用的是**梯度探针**：两个不同输入上梯度都非零才算。
  它可能把「只在特定输入下才被用到」的参数误判为未激活 —— 这类参数极少，
  且不影响本节的结构性结论。
- `EngramMemory.alive` 是 **buffer 不是 Parameter**，不在参数表里，
  所以不参与上面的元素数统计。它是 4096 个 bool，由语料统计决定。
- 本节**没有改动任何代码**，纯粹是清点。
