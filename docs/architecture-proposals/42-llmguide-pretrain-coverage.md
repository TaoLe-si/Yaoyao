# 42 · 基于 meko1/llm-interview-guide 预训练综述，对 Yaoyao 0.1.1 回滚版本建立完整训练逻辑

> 来源：[预训练目标与数据](https://meko1.github.io/llm-interview-guide/pretraining/pretrain)
> （原文缓存在 `docs/inbox_pretrain.md`，3,219 字节，98 行）
>
> **本文件规则**：所有声明都以 `origin/main` 当前 `HEAD = 7393d2c` 为准。
> 凡引用今天的设计（已回退），都明确标注 **(已回退)**。
> 凡文章谈到、当前回滚版本没有的部分，给出"如何补"的具体落点。

## 42.1 总体对应表

| 文章小节 | Yaoyao 回滚版本状态 | 本文件小节 |
|---|---|---|
| §一 什么是预训练 | ✓ 自监督 CLM，产出 base | 42.2 |
| §二 预训练目标（CLM / MLM / Span） | ✓ **仅 CLM**（不支持 MLM / Span） | 42.3 |
| §三 训练数据（清洗/去重/质量/配比/tokenize） | ✓ 完整实现（含 MinHash-LSH） | 42.4 |
| §四 训练范式与并行（DP/TP/PP/ZeRO/FSDP/BF16） | ✗ **只有单卡 + 8 GB GPU** | 42.5 |
| §五 训练稳定性（warmup/cosine/grad clip/spike） | △ warmup 硬编码 20，**无 cosine / grad clip / spike 处理** | 42.6 |
| §六 继续预训练 | △ 没有专门路径，但 `train_shards RESUME_DIR` 天然等价 | 42.7 |
| §七 高频追问 | 用 0.1.1 的真实文件一一对照回答 | 42.8 |

## 42.2 什么是预训练（§一）

文章原话：
> 预训练 = 在**海量无标注文本**上做**自监督学习**，让模型学到语言规律和世界知识，产出**基座模型**。

Yaoyao 0.1.1 完全匹配：

- 自监督 = 每个位置预测下一个 token 的交叉熵（`src/dual_state_cuda_loss.cuh:5` `ds_cross_entropy`）
- 海量无标注 = 流式分片读取 `shard_*.bin`，内存中只保留当前片（`src/train_shards.cu:60-68`）
- 产出 base = `<OUT_DIR>/final.dsb`（`train_shards.cu:212-213`），**只续写，不对话** —— 对应文章"base 模型只会续写，还不会对话；要经 SFT / 对齐才变成 Chat 模型"
- 训练入口：`build/train_shards.exe`（`scripts/build_train_shards.bat`）

## 42.3 训练目标（§二）

文章对照表（Yaoyao 落在 CLM 行）：

| 目标 | Yaoyao 是否支持 | 证据 |
|---|---|---|
| **CLM** 预测下一个 token | ✓ | `src/dual_state_cuda_loss.cuh:5` `ds_cross_entropy`；每位置用真实前文（teacher forcing） |
| MLM 还原遮盖 token | ✗ | 回滚版本没有 mask head；tokenizer 和 trainer 都假定全文监督 |
| PrefixLM / Span Corruption | ✗ | 同上 |

**CLM 的工程实现（回滚版本完整）**：

- **每位置单步交叉熵**（`src/dual_state_cuda_loss.cuh:6`）：
  ```cpp
  inline float seed_loss(Node output,unsigned target,bool supervised,float multiplier=1.f);
  ```
- **每位置是否计 loss** 由 `Token.loss` 决定（`src/bpe_pilot_reader.hpp`，TLP2 格式）；TLP3 全部 `loss=true`
- **teacher forcing**：`src/train_shards.cu:188-195` `replay.run(p, slots)` 把 `data.inputs` 当作真实前文传入；**不**用模型自己的预测
- **累计 loss 延迟收**：`DeferredLoss loss(1024)`（`train_shards.cu:104`），设备端累加，`loss.collect()` 拉回 host

**损失函数**（文章原文 $\mathcal{L} = -\frac{1}{T}\sum\log P_\theta(x_t\mid x_{<t})$）：

实际实现是 **token-level 归一** 的 NLL（`train_shards.cu:204`）：
```cpp
printf("UPDATE ... train_preupdate_NLL=%.6f", train_loss/targets, ...);
```
`train_loss/targets` = 监督 token 数上的平均 NLL —— 这是**逐 token**，与文章等价。

**Yaoyao 没有 MLM/Span，原因**：
文章也指出"主流 LLM 用 CLM"。Yaoyao 0.1.1 把整个架构押在自回归生成上，
回滚版本没有引入 BERT 风格的双向路径，是**与 Transformer 解耦的非自回归架构
之外**的另一种简化路线 —— 单一目标、少分支、训练稳定。

## 42.4 训练数据（§三）

### 42.4.1 数据来源（§3.1）

`scripts/build_corpus.cpp` 接受任意输入路径，逐个 tokenize 后统一进 TLP2/TLP3 分片。
回滚版本没有预定义"网页/代码/百科"的源类别分类 —— **来源靠用户在 stage 标签里手动
分组**（`build_corpus.cpp:77` 强制 `--stage core|mixed|interfere`）。

### 42.4.2 数据处理流水线（§3.2）—— **完整实现**

文章列的 6 步流水线，Yaoyao 在 `src/corpus_pipeline.hpp` 里有完整对应：

| 文章步骤 | Yaoyao 实现 | 源码位置 |
|---|---|---|
| 1. 清洗 | `--min-bytes`、`--max-repeat`、URL/HTML 清洗 | `corpus_pipeline.hpp:144,167-180`；`build_corpus.cpp:56-57` |
| 2. 去重 | **精确**（sha256）+ **近似**（MinHash-LSH 词 shingle） | `corpus_pipeline.hpp:6, 183-186, 505-518` |
| 3. 质量过滤 | 退化重复上限 + 最小字节 + 留出封存 | `corpus_pipeline.hpp:167-180`；`build_corpus.cpp:50, 55-57` |
| 4. 去毒/隐私 | 留口（`--seal FILE`），未自动检测有毒/PII | `build_corpus.cpp:49` |
| 5. 配比混合 | 三段课程表（`curriculum.tsv`） | `src/shard_corpus.cpp:103-110` |
| 6. Tokenize | 冻结 BPE-16123 数字强约束 | `src/bpe_train.cpp`（`is_digit` 守卫） |

**配比是 Yaoyao 的核心调参点**：curriculum.tsv 每行 `# stage shard_lo shard_hi steps_per_shard lr`。
默认 `core / mixed / interfere` 三等分，三档 LR 1e-3 / 8e-4 / 5e-4（`shard_corpus.cpp:106-108`）。

### 42.4.3 数据量级（§3.3）

回滚版本没有硬量级要求，但 `--docs-per-shard N`（`build_corpus.cpp:50`）决定每片文档数。
`shard_corpus.cpp:38` 范围 1..1e6。

文章给的量级："LLaMA 3 用 15T token" —— 这个数字**远超当前 Yaoyao 跑过的量级**
（参见 41.16 中的 581MB 实测）。这是回滚版本诚实的边界。

## 42.5 训练范式与并行（§四）—— **严重缺口**

文章列的 4 种并行：

| 类型 | Yaoyao 回滚版本 | 备注 |
|---|---|---|
| DP（数据并行） | ✗ | 单卡 |
| TP（张量并行） | ✗ | 单卡 |
| PP（流水线并行） | ✗ | 单卡 |
| ZeRO / FSDP | ✗ | Adam 状态驻**单机 host RAM**（`TAO_OPT_OFFLOAD=1` 替代分布式分片） |

`TAO_OPT_OFFLOAD` 的设计（已回退，原 `dual_state_sorted_trainer.cuh` 实现）：
"Adam 状态整体驻主机 RAM，训练步内零 CPU 参与" —— 这是把分布式 ZeRO 退化成单进程版的替代。

**BF16 / 混合精度**：
回滚版本**没有** BF16/FP16 训练。`dual_state_cuda_optimizer.cuh` 走的是 fp32 AdamW
（`ds_adamw` 内核签名 `(float* w, const float* g, float* m, float* v, ...)`）。

**梯度累积**：
`train_shards.cu:188-196` 8 个 round 累加 `targets` 一次性 `tr.update(targets, lr)`，
**本质就是梯度累积**：一次参数更新基于 8 × `replay.run` 的损失。

**梯度检查点（activation checkpointing）**：
`src/reusable_batch_graph.cuh` 用 CUDA Graph 捕获前向图 + 反向 tape。
**等价于**：把 `width` 个时间步的激活缓存在 device 上，反向时回放。
等价于"不做 activation checkpointing，全部留在显存"。

**约束与现实**：
回滚版本的现实是 **单卡 8 GB GPU**（见 `scripts/run_s1_real*.bat`），
并行根本没有意义。但这是 GPT-1 级（d=3200）的 2 层模型：
`docs/architecture-proposals/02-cpu-decode-throughput.md` 给出显存经验公式
`84.5·slots·width + 40·slots + 2121` MiB，把 slots×width 锁在 ~48。

**怎么补（按文章 §四）**：
1. **DP**：加 `torch.distributed` —— **不建议**，回滚版本没有 Python
2. **ZeRO-1（分 Adam 状态）**：把 Adam 状态按张量切到 N 张卡，每张卡只更新自己切片
3. **TP**：把 `d=3200` 按列切到 2 张卡，每张卡计算一半列；gather 后走 m-phase
4. **BF16**：把所有 CUDA kernel 的 fp32 改成 nv_bfloat16；主风险是 Adam `m`、`v` 在小 LR 下溢，建议保留 m/v 在 fp32，w/g 在 bf16
5. **梯度检查点**：把 `ReusableBatchGraph` 的 width 段分块，每块反向时重新前向 —— 能把显存 ~width× 降下来，代价是 ~30% 重算

## 42.6 训练稳定性（§五）—— **部分缺口**

文章列的稳定性措施：

| 措施 | Yaoyao 回滚版本 | 证据 / 缺失 |
|---|---|---|
| Warmup | △ 硬编码 20 步（`train_shards.cu:199`） | 不可配 |
| 余弦衰减 | ✗ | LR 训完 20 步后**常数不变** |
| 梯度裁剪 | ✗ | 没有 grad-norm 检查 |
| Loss spike 处理 | ✗ | 没有检测或回滚 |
| bf16 防溢出 | n/a | 全程 fp32 |

**当前实现**（`train_shards.cu:152-159, 199`）：
```cpp
float lr_max=0.001f;                                     // 默认 1e-3
// TAO_LR 可覆盖
float lr = tr.steps<20u ? lr_max*float(tr.steps+1u)/20.f : lr_max;  // 20 步线性 warmup
tr.update(targets, lr);
```

`tr.update` 在 `dual_state_sorted_trainer.cuh` 内：
```cpp
float norm=tr.update(targets, lr);                       // 返回的是当前梯度的 L2 norm
```
**这个 norm 是打印出来的，没有用作裁剪**（`train_shards.cu:204` `UPDATE ... norm=%.6f`）。
回滚版本**算出了 norm 但没用它**。

**怎么补（按文章 §五）**：
1. **余弦衰减**：
   ```cpp
   float lr = lr_max * 0.5f * (1.0f + cosf(M_PI * tr.steps / total_steps));
   ```
   总步数 = `shards.size() * steps_per_shard`
2. **梯度裁剪**：
   `if (norm > grad_clip) { scale_grads(grad_clip / norm); }`
3. **Loss spike 处理**：
   - 监控 `train_preupdate_NLL`，与最近 50 步均值比，>3σ 即 spike
   - spike 时回滚到上一个 `step_/final.dsb`，lr × 0.3，继续
4. **Min LR 地板**：lr_min = 1e-5（避免完全停下）

## 42.7 继续预训练（§六）—— **天然支持**

文章原话：
> 在已有基座上用特定领域语料继续做 CLM 训练，注入领域知识，再做 SFT。
> 关键是配比——混入部分通用数据防止灾难性遗忘。

Yaoyao 回滚版本的对应：

- **没有专门路径**，但 `train_shards RESUME_DIR`（`train_shards.cu:97-100`）
  + `tr.load_state(opt_state.bin)` 已经能"以同一基座再训练"
- **课程三阶段**（`shard_corpus.cpp:103-110`）本质就是"通用 → 通用+推理 → 全混"，
  在最终阶段换上一个代码/医疗语料目录，即等价于"继续预训练"
- **灾难性遗忘** 风险点：没有显式混入通用数据的机制，要靠用户自己安排课程表

**怎么补**：
`curriculum.tsv` 增加第 4 阶段 `domain`：前 5% 步混入 30% 通用语料，
让 domain 阶段不是"裸换语料"。

## 42.8 用回滚版本诚实回答文章 §七 的 7 个高频追问

### Q1：预训练为什么用自监督而不是监督学习？
`src/dual_state_cuda_loss.cuh` 只用 `target` 单 token，不接收任何标注。
`docs/architecture-proposals/20-curriculum-sharded-streaming-training.md` 论证了
0.9B 级语料无法整文件读入，必须分片流式 —— **前提就是无标注**。

### Q2：CLM / MLM / Span 的区别？主流用哪个？
回滚版本**只用 CLM**（42.3 节）。选 CLM 的理由与文章一致：
目标统一（每位置都有监督）、生成原生、不需要专门 mask head、训练简单。

### Q3：BERT (MLM) 和 GPT (CLM) 的本质区别？
回滚版本没有 BERT 路径。`docs/architecture-proposals/15-fuse-delta-mem-incompatibility.md`
等地方反复强调本架构是**非 Transformer** —— 选 CLM 不只是"用 GPT 不用 BERT"，
而是"用 dual-state 不用 attention"。

### Q4：为什么去重这么重要？
`src/corpus_pipeline.hpp:183-186` 显式实现：
> **近似去重（MinHash + LSH 分带）**：先按词切 shingle → 多个 hash 函数 → MinHash 签名 →
> 分带 → 带内 Jaccard 估计 → 阈值过滤（默认 0.8）
这正是文章点名的方法。

### Q5：数据配比为什么重要？
`src/shard_corpus.cpp:103-110` 三段课程表（core/mixed/interfere）就是数据配比的体现。
`--stage core|mixed|interfere`（`build_corpus.cpp:77`）让用户**手工**指定比例。

### Q6：预训练和微调的数据量差多少？
回滚版本**没有 SFT 路径**（已回退的 `grpo_rollout.cpp` 在 7393d2c 不存在）。
因此本仓库**只做预训练**，没有预训练 vs 微调的量级对比可言。
这是回滚版本的诚实边界。

### Q7：teacher forcing 是什么？有什么问题？
`src/train_shards.cu:188-195` 把真实前文一次性上 device（`replay.run`），
不进入 `hm.step` 的循环 —— 训练用的就是真实前文，没有"用自己的预测"暴露偏差。
回滚版本的 `exposure bias` 风险**与 Transformer 相同**：推理时自己的预测可能错，
训练时永远正确。

## 42.9 当前回滚版本的"训练逻辑"完整拓扑

```
原始语料
  │ build_corpus --out DIR --tokenizer TOK --stage core|mixed|interfere ...
  │   ├─ 精确去重（sha256）
  │   ├─ 近似去重（MinHash-LSH, 词 shingle, threshold=0.8）
  │   ├─ 质量过滤（min_bytes=64, max_repeat=0.35）
  │   └─ 封存（--seal FILE，可选，留出集来源）
  ▼
<DIR>/shard_00000.bin ... shard_NNNNN.bin  (TLP2 或 TLP3)
  │
  │ 同一可执行跑三次：
  │
  ├─ 第 1 段：train_shards ... core 400      → build/L1_pretrain
  ├─ 第 2 段：train_shards ... mixed 300 RESUME_DIR=build/L1_pretrain
  ├─ 第 3 段：train_shards ... interfere 300 RESUME_DIR=build/L1_pretrain
  │
  │ 单卡训练，8 GB GPU：
  │   ReusableBatchGraph(cudaGraph) replay
  │     └─ 每个 step：
  │         8 round × replay.run(p, slots)
  │         cross-entropy loss 累加 (DeferredLoss)
  │         AdamW 更新（fp32, 全程）
  │         norm 打印但 **不裁剪**
  │         lr = tr.steps<20 ? lr_max*(steps+1)/20 : lr_max
  │         if steps%50: 写 final.dsb + opt_state.bin
  │
  ▼
build/L1_pretrain/final.dsb  (fp32, 三值, 119M 参数在 d=3200)
build/L1_pretrain/opt_state.bin
  │
  ▼
评测 (eval_nll) —— 加载 fp32 dsb，CPU 串行/AVX2 推理
```

## 42.10 对照文章得出的"补全优先级"

按文章给出的 7 个能力点，回滚版本对应打分：

| 能力点 | 文章要求 | 回滚版本 | 优先级（要补的话） |
|---|---|---|---|
| CLM 目标 | ✓ 必须 | ✓ | – |
| 数据清洗 | ✓ 必须 | ✓ | – |
| 数据去重 | ✓ 必须 | ✓（精确 + MinHash） | – |
| 质量过滤 | ✓ 必须 | ✓ | – |
| 配比混合 | ✓ 必须 | △（手工三段） | 中 |
| 继续预训练 | △ 可选 | △（等价于 RESUME_DIR） | 低 |
| Warmup | ✓ 必须 | △（硬编码 20） | 中 |
| Cosine 衰减 | ✓ 必须 | ✗ | **高** |
| 梯度裁剪 | ✓ 必须 | ✗ | **高** |
| Loss spike | ✓ 必须 | ✗ | 中 |
| BF16 | △ 推荐 | ✗ | 中（但 d=3200 单卡装得下） |
| DP / TP / PP | 单卡不能 scale | ✗ | **推迟**（没有多卡） |
| ZeRO / FSDP | 单机可补 | ✗ | **推迟**（d=3200 单卡装得下） |
| 数据并行 | 单卡不能 scale | ✗ | 推迟 |

**最该补的两项**：余弦衰减 + 梯度裁剪。它们**没有结构性障碍**（当前代码已经在
算梯度 norm），改 5 行就能上。代价是 LR 不再"设了就跑"，要算总步数；
收益是稳定跑完千步以上不爆。
