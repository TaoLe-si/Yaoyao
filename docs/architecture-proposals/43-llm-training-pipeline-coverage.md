# 43 · 大模型训练 9 阶段全流程与 Yaoyao 0.1.1 回滚版本对齐

> 来源：[大模型训练全流程（从 0 到 1）](https://meko1.github.io/llm-interview-guide/pretraining/llm-training-pipeline)
> 原文缓存于 `docs/inbox_llm_pipeline.md`（已删除）。
>
> **基线 = 回滚版本（7393d2c，Yaoyao 0.1.1）**。
> 今天的新架构（d=3200 / 2-bit 打包 / m-phase AVX2 / 收敛门控 / SFT / GRPO 等）已被抛弃。
> 凡引用今天的设计，都标注 **(已回退)**。

## 43.1 文章的 9 阶段总览

```
① 数据工程 → ② 预训练(Base) → ③ 中期训练/退火 → ④ 有监督微调(SFT)
     → ⑤ 偏好对齐(RLHF/DPO/GRPO) → ⑥ 评估 → ⑦ 压缩量化
     → ⑧ 推理部署 → ⑨ 上线运营
```

## 43.2 Yaoyao 回滚版本在 9 阶段上的覆盖

| 阶段 | 回滚版本状态 | 关键证据 | 本文小节 |
|---|---|---|---|
① 数据工程 | ✓ 完整 | `corpus_pipeline.hpp` 6 步流水线（精确去重 + MinHash + 质量过滤 + 配比 + tokenize）；`bpe_train.cpp` 冻结 BPE-16123 + 数字强约束 | 43.3 |
② 预训练(Base) | ✓ 完整 | `train_shards.cu` + `dual_state_cuda_loss.cuh` + CLM + teacher forcing + 课程三段 | 43.4 |
③ 中期训练/退火 | △ 弱 | 有 `--max-repeat` 等质量过滤参数；**没有 LR cosine 退火**，**没有长上下文扩展位置编码** | 43.5 |
④ SFT | △ 数据 | 有 TLP2 对话格式；**但 trainer 只用它们当带监督的 TLP2 文本，不是真正 SFT** | 43.6 |
⑤ 对齐 (RLHF/DPO/GRPO) | ✗ 无 | 完全没有 reward / KL / critic / actor；`(已回退)` 的 `grpo_rollout.cpp` 不存在 | 43.7 |
⑥ 评估 | △ 弱 | 仅有 `eval_nll`（裸文本 NLL）；**没有 MMLU/GSM8K/HumanEval** 等基准 | 43.8 |
⑦ 压缩量化 | △ 训练端无 | 训练端全程 fp32（`ds_adamw(float*,...)`）；**解码端有** int8 + VNNI（`cpu_pipeline_rows.hpp`） | 43.9 |
⑧ 推理部署 | △ 弱 | `benchmark_noffn_s3.cpp` 测 TTFT / decode；**没有 vLLM/SGLang/TensorRT-LLM** | 43.10 |
⑨ 上线运营 | ✗ 无 | 没有监控 / 飞轮 / badcase 回灌 | 43.11 |

## 43.3 ① 数据工程（地基）—— ✓ 完整

文章原话：
> 数据质量决定模型上限。核心工作：采集与配比 / 清洗 / 去重 / 去污染 / Tokenization。

Yaoyao 实现（`src/corpus_pipeline.hpp`）：

| 文章步骤 | Yaoyao 实现 | 行号 |
|---|---|---|
| 采集 | `build_corpus INPUT...` 接受任意路径 | `build_corpus.cpp:62` |
| 配比 | 三段课程表 `core / mixed / interfere` | `shard_corpus.cpp:103-110` |
| 清洗 | `--min-bytes 64` `--max-repeat 0.35` | `corpus_pipeline.hpp:167-180` |
| 去重 | 精确（sha256）+ **近似（MinHash-LSH，词 shingle）** | `corpus_pipeline.hpp:183-186, 505-518` |
| 去污染 | `--seal FILE` 留口 | `build_corpus.cpp:49` |
| Tokenization | 冻结 BPE-16123 + 数字强约束 | `bpe_train.cpp:35` `is_digit` |

## 43.4 ② 预训练（Base Model）—— ✓ 完整

Yaoyao 实现：

- **目标**：CLM，`src/dual_state_cuda_loss.cuh:5` `ds_cross_entropy`
- **数据流**：`train_shards.cu` 流式分片 → `ReusableBatchGraph` cudaGraph replay → AdamW 更新
- **课程三段**：
  - 段 1：`build/train_shards.exe data/stage_core ... build/L1_pretrain 400`（LR=1e-3）
  - 段 2：... `300 32 32 RESUME_DIR=build/L1_pretrain`（LR=8e-4）
  - 段 3：... `300 RESUME_DIR=...`（LR=5e-4）
- **稳定性**：warmup=20 步硬编码（`train_shards.cu:199`），**无 cosine / 无 grad clip / 无 spike**
- **产物**：`<OUT_DIR>/final.dsb`（fp32 三值张量 + 7 uint32 config + tokenizer digest）

对齐文章的 4 个缺口：缩放定律、DP/TP/PP、ZeRO、BF16 —— 回滚版本都没有。

## 43.5 ③ 中期训练 / 退火 —— △ 弱

- **学习率退火**：**没有 cosine / linear / step decay**。课程三段 LR 切换（1e-3 → 8e-4 → 5e-4）发生在分片之间，不是分片内退火
- **长上下文扩展**：没有 RoPE / ALiBi / 位置编码外推（位置信息由 delta-memory 的 m 矩阵隐式承载）
- **能力增强**：`--stage core|mixed|interfere` 让人工分阶段，但**没有自动代码/数学比例加权**

怎么补：
1. `train_shards.cu` 加 `TAO_LR_DECAY`：从 step > decay_start 起按 `lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*progress))`
2. 课程表加第 4 阶段 `anneal`：所有数据上最后 5% 步用 LR=lr_min
3. 长上下文：先做 width=32→64 的 BPTT 长度扩展（不引入位置编码，靠 m 矩阵自然外推）

## 43.6 ④ 有监督微调（SFT）—— △ 数据格式有，训练路径无

Yaoyao 状态：

- **有 TLP2 对话格式**（`src/bpe_pilot_reader.hpp:14-26`）：
  - 文档布局：`BOS | USER | <正文, loss=false> | ASSISTANT | <正文, loss=true> | TURN_END | ...`
  - 角色 token：256/257/258/259（`language_data_contract.hpp`）
- **`corpus_pipeline.hpp:437-454`** 显式把对话写成 TLP2 记录

**关键发现：loss mask 已经是真 SFT 语义**

`bpe_pilot_reader.hpp:14-26` 在解析 TLP2 记录时：
```cpp
if(t[0].id!=BOS||t[0].loss) throw std::runtime_error("BOS");
// USER 段: 正文 loss=false
// ASSISTANT 段: 正文 loss=true
// USER 段的 TURN_END: loss=false（来自 m.assistant）
// ASSISTANT 段的 TURN_END: loss=true
```
`corpus_pipeline.hpp:442` 决定 TURN_END 的 loss：
```cpp
t.push_back({tao::data::TURN_END, m.assistant});
```
**意思是 USER 段不监督，ASSISTANT 段监督** —— 这**就是真 SFT 的 mask 语义**！

回滚版本的 SFT 实际上**已经能跑**，没人显式提过而已。

怎么补：
- 写 `scripts/run_sft.bat`：用对话语料走 `train_shards`，**不需要新代码**
- LoRA / QLoRA 没有 —— 三值权重与 LoRA 不兼容

## 43.7 ⑤ 偏好对齐（RLHF/DPO/GRPO）—— ✗ 无

Yaoyao 状态：**全部 0**。

- 没有 reward model
- 没有 critic / actor 双网络
- 没有 PPO 的 clip objective / GAE
- 没有 DPO 的偏好对 loss
- 没有 GRPO 的组内方差优势
- **`(已回退) train_grpo.cu` 与 `(已回退) grpo_rollout.cpp` 在 7393d2c 不存在**

为什么没有：回滚版本把"全栈对齐"放到 base 模型的逐字续写上。

GRPO 路线（按文章"当前热点"）最小实现：
```
for q in prompts:
  G = sample G times with temperature>0, top-k/p -> (text_i, reward_i)
  A_i = (reward_i - mean(reward)) / std(reward)   // 组内归一化
  loss = -E[ A_i * log P_theta(text_i) ] + beta * KL(theta || theta_ref)
```
这是 R1 风格。**当前回滚版本的生成是 greedy 贪心**，GRPO 需要**采样**（温度 > 0、top-k/p），是新增工作。

## 43.8 ⑥ 评估 —— △ 弱

- **`src/eval_nll.cpp`** —— 加载 `final.dsb`，按 `Token.loss` 计 NLL/ppl
  - 用途：衡量 base 模型的逐 token 预测能力
  - **不支持** MMLU/GSM8K/HumanEval 这类多选题/数值匹配/代码执行评测
- **数据污染防御**：`--seal FILE` 把留出封存进 corpus，eval 用相同文本就不污染
- **人类评估**：无

怎么补：
1. 多选题：`build/eval_mmlu.cpp`，加载 `MMLU/*.jsonl`，计算 `argmax P(选项)` 的准确率
2. 数学：`build/eval_gsm8k.cpp`，最后数字匹配
3. 代码：`build/eval_humaneval.cpp` + 沙箱执行（pass@k）
4. 中文：CMMLU、C-Eval 同上

**关键风险**：用 base 模型直接做 GSM8K 准确率必然接近 0 —— 没有指令遵循就不知道"问题是什么/答案是什么格式"。先做 ④ SFT（见 43.6）才能给 GSM8K 一个有意义的下界。

## 43.9 ⑦ 压缩量化 —— △ 训练端无，解码端有

- **训练端**：全程 fp32（`ds_adamw(float* w, const float* g, float* m, float* v, ...)`）
- **解码端**：
  - **三值权重**（{-1, 0, +1}）本身就是一种极致量化（2.32 bits/weight 理论值）
  - **int8 + VNNI**（`cpu_pipeline_rows.hpp:8, 15, 32`）：按矩阵规模分档
  - **AVX2 / AVX-512** 双路径（`bench_interleaved.cpp:60-62`）
  - **`cpu_compact_bundle.hpp`** 走 int8 紧凑打包
- **落盘**：fp32 在 `final.dsb`；量化发生在**加载到内存后**的解码阶段
- **蒸馏 / 剪枝**：无

为什么这样设计：8 GB GPU 装得下 119M × 4B = 476MB 的 fp32 权重，但**推理时的带宽**是瓶颈。三值 + int8 VNNI 把内存流量压到 ~25%（int8）甚至 12.5%（三值）。

W4A16 / FP8 没实现：
- W4A16：三值已经 2-bit，再压收益小
- FP8：三值投影损失 fp32 精度做 FP8 反而变差；不在路径上

## 43.10 ⑧ 推理部署 —— △ 弱

- **`benchmark_noffn_s3.cpp`** 已经测 TTFT（`ttft`）、mean_reply_ms、decode_steps（`benchmark_noffn_s3.cpp:115-119`）—— **指标齐了，但只测一个回合**
- **连续批处理**：单卡单请求，没有 concurrent batching
- **PagedAttention**：不需要（Yaoyao 没有 KV cache，没有 attention）
- **投机解码**：无
- **没有 vLLM / SGLang / TensorRT-LLM**：这些是 GPU attention 框架，Yaoyao 是 CPU/GPU 混合 + 状态机推理，框架对接不上

怎么补：
1. **HTTP server**：`scripts/run_serve.ps1` 用 `httplib` 写 30 行做 OpenAI 兼容 endpoint
2. **批处理**：在同一进程内并发处理 N 个 slot，每个 slot 走 greedy_pipeline_grouped_model
3. **量化开关**：CLI `--int8`、`--vnni`、`--avx512`、`--fast-act` 已有
4. **指标**：扩展 `benchmark_noffn_s3` 测 p50/p95/p99 latency 与并发吞吐

**国产算力（昇腾 MindIE）**：无适配。诚实边界。

## 43.11 ⑨ 上线运营 —— ✗ 无

- 没有 online eval
- 没有 badcase 回灌
- 没有 drift 检测
- 没有红队 / 护栏

这是基础设施，不是模型工作。回滚版本是**模型研究仓库**，不是产品。

## 43.12 用回滚版本诚实回答文章的"高频追问"

### Q1：完整讲一下从 0 训练大模型的流程？
Yaoyao 仓库只覆盖 ①②。③-⑨ 全部缺失或弱。

### Q2：预训练、SFT、对齐分别学到什么？
Yaoyao 只做 ② 预训练。SFT 在数据层已能跑，对齐完全没有。

### Q3：知识应该在哪一步注入？
Yaoyao 没有 RAG 路径 —— 全部知识靠预训练注入。这是非 Transformer 架构的特点：没有 KV cache，没有 in-context 检索增强。

### Q4：RLHF、DPO、GRPO 什么关系？
Yaoyao 三者都没有。GRPO 是"当前热点"，`(已回退)` 的 `train_grpo.cu` 曾经在仓库里。回滚决定没有附 message —— "严重问题"是不是包括 GRPO 实现？**未知**，但既然整个今天的新架构都回退了，GRPO 也在内。

### Q5：缩放定律在流程里起什么作用？
回滚版本**没有** Chinchilla 自动预算分配。当前配置 d=3200, L=2 的 119M 模型是手工定的。

### Q6：大部分团队会做哪几步？
回滚版本适合"自建 base"的极小团队，主流路径（**用开源 base + SFT/LoRA + RAG/Agent**）只有 SFT 部分能跑，缺 LoRA、缺 RAG、缺 Agent。

### Q7：评估为什么要贯穿全程？
回滚版本的评估只有 `eval_nll`（NLL/ppl），没有 MMLU/GSM8K/HumanEval。**诚实边界**：基础指标能算，**业务指标全缺**。

## 43.13 回滚版本训练逻辑完整拓扑（9 阶段视角）

```
① 数据工程  ✓ build_corpus + corpus_pipeline + bpe_train
       │     精确去重 + MinHash-LSH + 质量过滤 + 课程配比
       ▼
② 预训练     ✓ train_shards (CLM, 三段课程, fp32, 单卡, warmup=20)
       │     final.dsb + opt_state.bin
       ▼
③ 中期/退火  △ LR 在分片间切换（1e-3/8e-4/5e-4），分片内恒定
       │     无 cosine / 无长上下文扩展 / 无显式 anneal 阶段
       ▼
④ SFT        △ TLP2 数据结构有，但要走 train_shards 当普通 CLM 训
       │     无独立 SFT 路径，无 LoRA/QLoRA
       ▼
⑤ 对齐       ✗ 无 reward model / 无 critic / 无 KL
       │     (已回退) 的 train_grpo.cu / grpo_rollout.cpp 已消失
       ▼
⑥ 评估       △ eval_nll 测 NLL/ppl
       │     无 MMLU/GSM8K/HumanEval 基准
       ▼
⑦ 压缩量化   △ 训练 fp32；解码 int8/VNNI/三值 + AVX2/AVX-512
       │     无蒸馏、无剪枝
       ▼
⑧ 推理部署   △ benchmark_noffn_s3 测 TTFT/decode
       │     无 HTTP server、无连续批处理、无 vLLM/SGLang/TensorRT-LLM
       ▼
⑨ 上线运营   ✗ 无监控、无飞轮、无 badcase 回灌、无护栏
```

## 43.14 补全优先级（按文章给出的标准流程）

| 优先级 | 阶段 | 缺什么 | 改动量 |
|---|---|---|---|
| **高** | ③ 退火 | LR cosine decay | +5 行 |
| **高** | ③ 退火 | grad clip | +1 行（norm 已算） |
| **高** | ④ SFT | 写一个 `run_sft.bat` 跑 TLP2 数据 | +1 个脚本 |
| 中 | ⑥ 评估 | MMLU/GSM8K 评测器 | +2 个 .cpp |
| 中 | ⑧ 部署 | HTTP server（OpenAI 兼容 endpoint） | +50 行 |
| 中 | ⑤ 对齐 | GRPO 最小实现 | +500 行（重写 SFT + 加采样） |
| 低 | ⑤ 对齐 | DPO / PPO | +800 行 |
| 低 | ⑦ 量化 | W4A16（已有三值收益很小） | 实验性 |
| 低 | ⑨ 运营 | 监控 / 飞轮 / 护栏 | +1000 行 + 部署 |

**最该补的前 3 项**：

1. LR cosine 退火（5 行代码，立刻提升稳定性）
2. grad clip（1 行代码）
3. 写一个 `run_sft.bat` 让 TLP2 数据走 train_shards（1 个脚本，立刻把 base 转成 assistant）

这三项**完全没有结构性障碍**，都是回滚版本已经具备能力的暴露，
而 ⑤ 对齐和 ⑨ 运营是上层工作，受限于当前 base 模型质量。
