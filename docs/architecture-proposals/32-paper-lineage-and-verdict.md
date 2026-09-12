# 32 · 六篇论文的谱系定位与架构裁决（含对 doc 31 的更正）

> 2026-09-12。由用户提供 6 篇论文（`C:\Users\Administrator\Desktop\论文`）驱动，六篇全部提取为文本
> （`docs/papers/*.txt`）并逐篇深读。本文回答：**在我们这个「非 Transformer」的谱系里，
> 哪些技术对症、哪些不适用。**

## 零、我们的谱系位置（先把坐标系定死）

实测事实（非推断）：

| 性质 | TaoVm 实测值 |
|---|---|
| 算子身份 | `dual-state-4-noffn-delta-mem-input-sqrt-d`（读自 `final.dsb` 头部）|
| 注意力 | **无**。`src/` 全文搜 `softmax\|attention\|attn\|kv_cache\|qkv` → **命中 0 条** |
| 状态 | s∈R^128 向量 + M∈R^{512×64} 矩阵，**定长 257 KiB，与序列长度无关** |
| 读出 | `o = M q`（**带 query**，见 §三）|
| 写入 | delta 规则 `M ← M + β(v − Mk̂)k̂ᵀ`，k̂ 已 L2 归一化 |
| 对照 | Transformer 的 KV cache 随 n 线性增长：n=4096 时约 33.6 MiB，是我们的 **128 倍** |

所以六篇里只有五篇是**同门（线性 RNN / SSM / 门控递归 / 联想记忆）**，Transformer 是**对照基线**：

```
同门谱系:  LSTM(1997 CEC+门) → xLSTM(sLSTM 记忆混合 / mLSTM 矩阵记忆+指数门+归一化)
                              → Mamba(输入相关选择性 Δ) → TFLA(数值稳定内核)
我们在此:  delta 规则联想记忆 M + 门控向量状态 s   ← 位于 xLSTM 的 mLSTM 与 DeltaNet 交叉点
对照基线:  Attention(Transformer) — 非 Transformer，仅借 1/√d_k 与残差/norm 放置
正交参考:  DeepSeek-V4.1-Flash — Transformer MoE，架构不适用，仅训练配方可迁移
```

## 一、五篇同门论文的独立裁决（收敛结论）

### 1.1 没有任何一篇能解决 P1（灾难性遗忘）—— 5/5 一致

这是本轮最重要的收敛结论。逐篇证据：

| 论文 | 门/衰减作用于 | 有无遗忘实验 | 判定 |
|---|---|---|---|
| LSTM 1997 | CEC 保护**误差流**，不保护**权重** | 无（全是单任务、序列间重置）| 不能 |
| xLSTM | `f_t ⊙ C_{t-1}` = **序列内状态**衰减 | 无（全文无域切换）| 不能 |
| Mamba | `Ā = exp(ΔA)` 同上，原文 line 398-403 明说是"序列内" | 无 | 不能 |
| TFLA | kernel 论文，不碰数据顺序 | 无 | 不能 |
| DeepSeek-V4.1 | 全 45T token **混合**分布，非阶段切换 | 全文 **0 次** forget/continual/rehearsal | 不能 |

**机制**：`f_t`、`Δ`、CEC 全都只调节「递归状态」的衰减，而遗忘发生在**权重**被新域梯度覆盖。
两者是不同的量。所以「加遗忘门」不能治 P1 —— 这恰好与我在 doc 31 的实测吻合：
我们的 s 分支**本来就带 sigmoid 门**（`s += σ(a)(tanh(u)−s)`），却仍遗忘 +15.2%。

**DeepSeek-V4.1 反而正面支持 doc 31 的诊断**：它在 45T token 上全程用**固定配比混合流**
+ 余弦衰减（warmup 2000 步 → 2.6e-4，28T 后衰减到 2.6e-5）。我们现在的做法
（25 片互不重叠、纯顺序、恒定 5e-5）**正是它刻意规避的非平稳配方**。

### 1.2 P2（216 KB 计算容量）只能靠「把容量搬到状态」，不能靠权重

| 论文 | 手段 | 关键数字 |
|---|---|---|
| Mamba | 状态维 N 展开 | Table 10（line 914-923）：N 1→16 只 **+1% 参数**换 >1.0 ppl，**但前提是 B,C 选择性** |
| xLSTM | `C ∈ R^{d×d}` 矩阵记忆 | 明确是**无参数状态**；350M 上 26.01 → 17.70(指数门) → 13.48(矩阵记忆) |
| DeepSeek | Engram 稀疏查表 | 196B 参数解耦出计算路径（强依赖 552B 规模，**不适用**）|

**可用的杠杆**：我们的 s 和 M 是**运行时激活**，不进 9,502,720 ternary 预算。
扩大状态（dk 64→128、多槽位）**不消耗权重容量**，只消耗显存与 MAC。
但 Mamba Table 10 同时警告：**只扩状态维、不给选择性读出，增益 <0.1**。

### 1.3 P3（读出）—— 已经解决了一半，且论文们证实我们方向正确

**必须更正 doc 31 的一处错误**（我已回改 doc 31 §五）：

> doc 10 说「读出 `r = W_rs·s + W_rm·m` 无 query，信息在状态里取不出来」。
> **那是旧算子 `dual-state-3`。** 交付模型 `s2_night1/final.dsb` 的算子身份是
> `dual-state-4-noffn-delta-mem-input-sqrt-d`，schema 含 `mem.key/query/value`，
> 读出实为 **`o = M q`，本来就带 query**。

而 Mamba / xLSTM / TFLA **三篇独立地**把「读出必须含 query」列为核心贡献：

- Mamba：`C_t = s_C(x_t)` ——「把 C 钉死成常数」正是 LTI 缺陷（原文 line 214-218）
- xLSTM：`h = C_t q_t / max(|n_t^T q_t|, 1)`，显式 query
- TFLA：`h̃ = C_t^T (q_t/√dqk)`

**→ 我们的 delta-mem 已经实现了它们的推荐。** doc 16 也实测过：换到 3,787 个名字的 L4 语料后，
delta-mem 把名字抄对率从 ds3 的 **1/12 提到 9/12**（姓氏 12/12）—— 这是本项目第一个架构级正向结果。

### 1.4 P4（计数/终止）—— 文献公认的未解问题，不是我们的缺陷

> **LSTM 1997 §6 Limitations 原文**：
> "All gradient-based approaches, however, suffer from practical inability to precisely count
> discrete time steps. If it makes a difference whether a certain signal occurred 99 or 100 steps
> ago, then an additional counting mechanism seems necessary."

LSTM 的作者在**1997 年**就明说计数需要外挂机制，且没给设计。xLSTM、Mamba、TFLA、DeepSeek
四篇至今都没有提供。**所以 P4 是开放的文献问题，不是我们实现得差。**

## 二、五篇合起来才看得出的关键发现：我们的 M 缺一个遗忘门

这一点单看任何一篇都看不出来，是我把论文与**内核源码**对照后才确认的：

```
我们的内核 delta_mem_kernels.cuh (dm_forward, L40-54):
    acc = Σ_j M[i,j]·k̂[j]
    g   = β·(v[i] − acc)
    M[i,j] ← src[j] + g·k̂[j]        ← 只有累加，没有任何衰减项
    o[i] = Σ_j M[i,j]·q[j]           ← 写后读

xLSTM Eq.19-27:
    C_t = f_t⊙C_{t-1} + i_t·v_t k_t^T     ← f_t 是遗忘门
    n_t = f_t⊙n_{t-1} + i_t·k_t           ← normalizer 状态
    h_t = o_t⊙( C_t q_t / max(|n_t^T q_t|, 1) )

Mamba Theorem 1 (line 370-374):
    g_t = σ(Linear(x_t));  h_t = (1−g_t)h_{t-1} + g_t x_t   ← 选择性 = 遗忘门
```

**我们缺两样，但主次分明**：

1. **遗忘门 `f_t`（主要）**：doc 16 已指出，单层只有一个 M，第二组关联与第一组共享 key 方向，
   「delta 更新互相抵消」→ 颜色 LCS = 0.00（**低于随机 0.17**）。
   加 `f ⊙ M` 衰减是文献一致的做法（xLSTM、Mamba 都有）。
2. **normalizer `n_t`（次要，且有反证）**：TFLA line 2140-2143 明确说 mLSTMsig **默认 n=1**，
   「加 normalizer 不改变 transfer behavior」；normalizer 只对**指数门** mLSTMexp 才是必需的（line 2130）。
   所以 n 不是优先项 —— 子代理 TFLA 那条主张在此被论文原文纠正。

## 三、优先级排序的实验计划（预登记判据）

### E1（最高优先，零架构改动）排练 + LR 余弦衰减

- **依据**：5/5 论文一致（P1 只能靠数据/优化配方）；DeepSeek-V4.1 的混合流 + 余弦衰减。
- **做法**：把「25 片纯顺序、互不重叠」改成**每个新域分片混入约 20% 旧域数据**；
  末段加余弦衰减（5e-5 → 5e-6）。**不改架构、不改算子、不需重训。**
  我们的 `steps_per_shard` 是全局量，所以「每 k 片插 1 片旧数据」用**分片目录的硬链接顺序**即可实现。
- **判据**：wiki NLL 回退从 **+15.2% 压到 <+8%**，同时 reason/alpaca 增益保留 ≥90%。
  若仍 >+12% → 数据混合不足以治 P1，需转向权重正则（EWC）或冻结部分容量。

### E2（次优先，小架构改动）给 M 加遗忘门

- **依据**：§二 —— 内核实测无衰减项；xLSTM/Mamba 一致要求 f_t。
- **做法**：`f = σ(W_f x + b_f) ∈ R^dk`（dk=64），`M ← f⊙M + β(v − (f⊙M)k̂)k̂ᵀ`，
  `b_f` 初值取 +3~6（xLSTM Table 2 实测：取 +∞ 会 NaN）。
  新增 `W_f: 64×512 × 2 层 = 65,536 ternary`，占 1,114,112 的 **+5.9%**。代价很小。
- **判据**：doc 16 的交叉注入颜色命中率从 **0/12 升到 ≥4/12**（名字不退化）；
  且 M 的 Frobenius 范数在 1000 步分片内不再单调增长。

### E3（第三）dk 64→128 或双槽位

- **依据**：Mamba Table 10（状态维 +1% 参数换 >1.0 ppl，**但需配选择性读出**）；
  doc 16 §六明确列出「加大 dk 给多关联留地址空间」为待做项。
- **做法**：dk 64→128。新增 `mem.key + mem.query` = **65,536 ternary（+5.9%）**；
  状态 M 由 256 KiB → **512 KiB**（运行时，不进权重预算）。
- **判据**：颜色命中率进一步提升；若名字/颜色均无改善 → 瓶颈不在地址空间。

## 四、明确**不适用**的技术（避免浪费工期）

| 技术 | 出处 | 为何不适用 |
|---|---|---|
| KV cache 压缩 / MLA / FP4 KV | DeepSeek-V4.1 | 我们**没有 KV cache**（状态定长 257 KiB）|
| SWA Bounded Replay / 稀疏 indexer | DeepSeek-V4.1 | 依赖注意力窗口与 KV 池 |
| MoE 路由 | DeepSeek-V4.1 | 1,114,112 计算权重无法专家化 |
| CED 跨层复用 | DeepSeek-V4.1 | 论文是 20+20 层每 4-6 层复用；**L=2 时只剩 1 层状态深度** |
| TFLA 分块并行 / Flash Linear Attention 内核 | TFLA | 收益在 H100 的 HBM 流量与 tensor core；2 层 d=512 无 tensor core。**且论文自己记录 xl_chunk 有不可恢复 spike** |
| chunkwise 并行训练公式 | TFLA | 我们逐 token 自回归训练，引入只增复杂度与数值风险 |
| 逐元素对角门控读出 (`o_j = c_j·m_j`) | Mamba App. A | 论文点名：只是 GLU，是 trivial transformation，**不构成内容寻址** |
| 位置编码 / softmax 注意力 | Attention | 非 Transformer，无注意力 |
| EWC / 权重正则 | （六篇均无）| 文献缺口，需自行设计 |

**唯一从 Attention 可借的**：`1/√d_k` 缩放（我们已有 `input-sqrt-d`）与 `LayerNorm(x + Sublayer(x))` 残差/norm 放置。

## 五、交付物

- 六篇论文提取文本：`docs/papers/{Attention_is_all_you_need, DeepSeek_V41_Tech_Report, Mamba, TFLA_重要, lstm_重要, xLSTM_重要}.txt`
- 本文：`docs/architecture-proposals/32-paper-lineage-and-verdict.md`
- 更正后的 doc 31：`docs/architecture-proposals/31-forgetting-architecture-verdict.md`（§五 已标注更正）

## 六、本轮的方法论教训

1. **子代理的结论必须回原文核验**。TFLA 子代理称「必须加 normalizer」，论文 line 2140-2143
   明确说 mLSTMsig 默认 `n=1` 且不影响 transfer —— **过度主张，已纠正**。
   Mamba 子代理的 Theorem 1、Table 10、74K 参数三个数字我逐一到原文核验，**全部准确**。
2. **论文结论必须对照内核源码**。§二 的「缺遗忘门」不是任何一篇论文直接说的，
   是把 xLSTM 的 `f_t` 与我们的 `dm_forward` 内核逐行对照才发现的。
3. **旧的诊断文档会过期**。doc 10 的「读出无 query」在算子升级到 delta-mem 后已失效，
   但被 doc 31 引用了一次。**引用旧结论前必须先核对交付产物的算子身份。**
