# 40 · 回滚版本（7393d2c，Yaoyao 0.1.1）的完整训练架构

> 引用规则：所有行号与符号均以 `origin/main` 当前 `HEAD = 7393d2c` 为准。
> 不引用今天的设计 —— 今天的改动已回退，分支备份在
> `backup-before-rollback-20260912-195936`。

## 40.1 顶层定位

Yaoyao 是**非 Transformer** 的双状态 LLM 架构：

- **三值权重** `{-1, 0, +1}`，**不打包**到 2-bit（在 0.1.1 中），全部用 fp32 表示
  （参见 `src/cpu_pipeline_rows.hpp`、`src/cpu_ternary_avx2.hpp`）；
- **双状态**：每层维护 `s`（状态向量，d=128）与 `m`（增量记忆矩阵，d×dk），
  见 `src/dual_state_config.hpp`、`src/delta_mem_kernels.cuh`；
- **没有 FFN**（`src/train_shards.cu:1` `#define TAO_NO_FFN`，硬编码而非编译选项）；
- **没有因果注意力**，记忆由 delta rule 更新；
- **训练与推理都是 CPU/GPU 协同**：CUDA 训练，CPU 解码；这是从训练器到 rollout
  的**默认路径**，没有"只用 CPU"或"只用 GPU"的训练。

## 40.2 模型形状（`src/dual_state_config.hpp`）

```cpp
struct Config {
    uint32_t layers=2, d=512, s=128, m=512, e=1024, vocab=16384;
    uint32_t dk=64;
};
```

- `schema(c)` 在 `dual_state_config.hpp:19-30` 定义了完整的张量集合，包括
  `mem.key(mem.query/mem.value/mem.beta/mem.beta.bias)`。
- 校验 `cfg.validate()`: `dk <= d`、`layers <= 64`、`bundle dim <= 8192`、
  `vocab <= 262144`。
- **默认形状**：L=2, d=512, s=128, m=512, e=1024, vocab=16384, dk=64。
  这是**没有缩放**到 GPT-1 级别的最小工作配置。
- 显存与吞吐经验公式见 `docs/architecture-proposals/02-cpu-decode-throughput.md`。

## 40.3 Tokenizer

**冻结分词器**，BPE-16123 训练 + 数字强约束：

- **训练器**：`src/bpe_train.cpp` —— byte-level BPE，
  **约束**：任何 merge 都不能含 ASCII 数字字节 (0x30-0x39)，
  "123" 始终编码为 `[1][2][3]`，保留进位与位值结构。
- **加载/校验**：`src/byte_bpe.hpp` `ByteBpe::validate()`，要求 `merges.size() <= 16123`。
- **训练器强制**：`src/train_shards.cu:81` —— 不设 `TAO_ALLOW_TOKENIZER` 时，
  校验 digest == `18b1c761bbc13d7f29ff99db86b3eb17502d7ba2afe45c549fc4381fffc037fa`。
- **vocab 大小**：`256 (bytes) + 4 (specials: 256/257/258/259, BOS/USER/ASSISTANT/TURN_END)
  + 16123 (merges) + 1 (SENT) = 16384`，与 `Config::vocab=16384` 对齐。
- 详见 `docs/EXPERIMENT_EVIDENCE_0.1.md` 与 `docs/LEARNING_MODE_0.1.md`。

## 40.4 语料格式

两种格式共存，训练器**自动按 magic 识别**（`src/train_shards.cu:36-38`）：

### TLP2（对话）
`docs/architecture-proposals/04-engram-conditional-memory.md` 与
`src/bpe_pilot_reader.hpp` 描述。
`read_bpe_pilot` 返回带 role 标签的 `vector<vector<Token>>`，正文 token 的 loss 由 role 决定。

### TLP3（纯文本/代码）
`src/plain_lm_format.hpp` 全套定义（**唯一文件**，不可改）：
- 文档布局：`BOS(256) + [文本 token, loss=true] + TURN_END(259, loss=true)`。
- 文件头：`"TLP3" + 64 字节 tokenizer digest = 68 字节`（`plain_lm_format.hpp:29-32`）。
- 自检约束：`front().id==BOS && back().id==TURN_END && size>=3`
  （`plain_lm_format.hpp:9`）。
- 流式读取：`Tlp3ShardReader`（`plain_lm_format.hpp:77-87`）—— 一次最多 `max_docs` 篇。
- **TLP2 / TLP3 的角色 token 在训练中互不替代**。TLP3 显式拒绝 role 序列
  （`plain_lm_format.hpp:4-7`）。

## 40.5 分片流式训练器（核心入口）

**唯一的多分片训练器**：`src/train_shards.cu`。

### 命令行
```
train_shards SHARD_DIR TOKENIZER.bbp OUT_DIR STEPS_PER_SHARD
            [SLOTS WIDTH] [RESUME_DIR] [START_SHARD]
```
（`train_shards.cu:15, 45-46`）

### 参数约束（`train_shards.cu:55-57`）
- `steps_per_shard`: 1..1e6
- `slot_count`: 1..256（**默认 32**）
- `graph_width`: **8..512**（**默认 32**，硬下界 8 —— 不可 < 8）

### 三段式课程
`docs/architecture-proposals/20-curriculum-sharded-streaming-training.md` 定义：

- `core`   —— 自然语言 + 推理基础（前 1/3 分片，steps_per_shard=400，lr=0.001）
- `mixed`  —— 自然语言 + 推理 + 代码（中 1/3 分片，steps_per_shard=300，lr=0.0008）
- `interfere` —— 全部数据混训（后 1/3 分片，steps_per_shard=300，lr=0.0005）

课程表模板由 `src/shard_corpus.cpp:103-110` 自动生成到 `<OUT_DIR>/curriculum.tsv`。

### 三个阶段的运行方式
**同一个可执行跑三次，每次换 `RESUME_DIR`**（`train_shards.cu:11-13, 97-100`）：

```
# 阶段一：core
train_shards data/stage_core build/tok_v2.bbp build/L1_pretrain 400

# 阶段二：从阶段一续训，mixed 步数 = 300
train_shards data/stage_mixed build/tok_v2.bbp build/L1_pretrain 300 32 32 build/L1_pretrain

# 阶段三：interfere
train_shards data/stage_interfere build/tok_v2.bbp build/L1_pretrain 300 32 32 build/L1_pretrain
```

### 数据流
```
fs::directory_iterator(shard_dir) → sort by name → for si in [start_shard..n):
  read_shard() ──magic──┬── "TLP2" → read_bpe_pilot
                         └── "TLP3" → read_tlp3
  PilotCursor(docs, slot_count, false)
  ShuffledEpochCursor(PilotCursor, seed_base+si, identity)  // train_shards.cu:175
  for u in 0..steps_per_shard:
    for r in 0..8:                              // 8 个 round / step
      if epoch.exhausted(): begin_next_epoch(true)   // train_shards.cu:189
      take_batch(epoch, graph_width)                // batched_slot_plan.hpp:6
      replay.run(p, slots)                          // reusable_batch_graph.cuh:16
      accumulate_loss(loss.total, loss.bad)
    lr = steps<20 ? lr_max*(steps+1)/20 : lr_max  // warmup=20
    tr.update(targets, lr)
    if tr.steps%50 == 0: export_ckpt(false)
  export_ckpt(false)   // 每片结束
```

### 学习率
- 默认 `lr_max=0.001f`（`train_shards.cu:152`）
- `TAO_LR` 可覆盖（`train_shards.cu:153-159`），范围 1e-6..0.1
- **warmup 硬编码 20 步**（`train_shards.cu:199`），线性 `(steps+1)/20`
- **无 LR 衰减**（回滚版本不支持 decay）

### 优化器状态（Adam）
`src/dual_state_sorted_trainer.cuh` `SortedGpuTrainer::save_state/load_state`：

- `opt_state.bin` 写盘时机：
  - 每个检查点（每 50 步）后无条件写（`train_shards.cu:127-130`）
  - 关闭：`TAO_OPT_STATE=0`（`train_shards.cu:110`）
- 文件：`build/L1_pretrain/opt_state.bin`，跨分片持续，跨课程阶段持续
- **原地续训**：`train_shards.cu:71` 强制 `!fs::exists(out)`，**禁止原地续训**，
  必须复制 `opt_state.bin` 到另一个 `OUT_DIR`
  （`backup-before-rollback-20260912-195936` 才有原地续训能力 —— 已回退）

### 检查点
每 50 步一次（`train_shards.cu:206`），产物：

```
<OUT_DIR>/step_<N>/final.dsb          # 用 CpuModel.copy + save_bundle 写盘
<OUT_DIR>/opt_state.bin               # Adam 状态
<OUT_DIR>.log                         # stdout/stderr 通过 freopen
```

**dsb 模型格式**：`src/dual_model_stream.hpp` `save_model_stream`：
- 写 `Config` 的 7 个 uint32（layers/d/s/m/e/vocab/dk）+ 张量名表 + float 数据
- `load_model_stream` 是逆向
- 校验：`c.layers>64 || c.d>8192 || c.s>8192 || c.m>8192 || c.e>32768 || c.vocab>262144` 直接拒
- 注意：**没有 fp16/2-bit/ternary quantization**，权重全部以 fp32 落盘
  —— 推理时直接把 fp32 三值当 fp32 用

### 终止控制
`OUT_DIR.stop` 文件存在即停（`train_shards.cu:181-184`）：
```
if fs::exists(stoppath):
  if tr.steps%50: export_ckpt(false)   // 不是 50 倍数时也存一次
  return 0
```

### 进度打印
```
UPDATE step=%u shard=%u positions=%zu targets=%zu train_preupdate_NLL=%.6f lr=%.6f norm=%.6f
```
无 stage timing，无收敛判据，无 NLL 留出评估（这些是今天版本独有的）。

## 40.6 训练核心图

`src/train_yaoyao_graph_gpuval.cu`（被 `train_shards.cu:17` 包含）：

- `SortedGpuTrainer` —— 排序后的 Adam 状态 + 张量集合
- `ReusableBatchGraph(slots, width)` —— CUDA Graph 捕获，replay 复用
  （`reusable_batch_graph.cuh:9-14`），`.run()` 把 `padded_batch_plan` 上传到
  device 再 `cudaGraphLaunch`（`reusable_batch_graph.cuh:18-26`）
- `SequenceSlots(tr, slot_count)` —— 跨 step 的状态容器（s/m 各 `slot_count` 层）
- `DeferredLoss(1024)` —— loss 在 device 累积，`loss.collect()` 拉回 host
- `accumulate_loss<<<1,1>>>` —— `train_shards.cu:193`，GPU→host 单值

### 张量布局（`dual_state_config.hpp:19-30`）
```
layer.{i}.s.candidate.x    s × d     (ternary)
layer.{i}.s.gate.x         s × d     (ternary)
layer.{i}.s.candidate.bias  1 × d
layer.{i}.s.gate.bias      1 × d
layer.{i}.s.norm.weight    1 × d
layer.{i}.mem.key          dk × d    (ternary)
layer.{i}.mem.query        dk × d    (ternary)
layer.{i}.mem.value        m × d     (ternary)
layer.{i}.mem.beta         1 × d
layer.{i}.mem.beta.bias    1 × 1
embedding                   vocab × d (ternary)
readout.w                   vocab × d (ternary)
readout.b                   1 × vocab
```

## 40.7 推理 / 生成

**没有专门的生成器**。回滚版本的"问答"是直接读 `final.dsb` 进 `CpuModel`，再调
`GreedyPipelineGroupedModel` 做贪心解码（`src/greedy_pipeline_grouped_model.hpp`）。

- 这是 `src/eval_nll.cpp` 同样的路径，**没有采样温度、top-k、top-p、重复惩罚**
  等参数（这些是今天加进 `grpo_rollout` 的）
- **没有 SFT 路径** —— 模型只能续写 TLP3 文档，不能回答结构化问题

## 40.8 评测

`src/eval_nll.cpp`：

- 加载 `final.dsb`，遍历文本，按 `Token.loss` 计 NLL
- 输出 `NLL=... ppl=... doc_nll=...`，**没有留出集自动切分**、
  **没有帧格式开关**、**没有 SFT-QA 评测**（这些是今天加进来的）
- 详见 `scripts/run_smoke.bat` / `scripts/run_real.bat`

## 40.9 反推这一版的局限

- **没有 SFT**：Q&A 能力完全靠 base model 在 TLP3 续写任务上学到的"运气"
- **没有 GRPO**：`src/grpo_rollout.cpp` 在回滚后**不存在**
- **没有重复惩罚**：`30117ba`（decoder repetition penalty）在
  7393d2c **之后**才提交 —— 已回退
- **没有 m-phase AVX2 向量化**：`3a302cd` 已回退
- **没有 VNNI 去重**：`0e21f56` 已回退
- **没有 2-bit packed weights**：`ba298a6` 已回退
- **没有原地续训守卫**：`331dda1` 已回退
- **没有 NLL 留出统计**：今天 `CONV_PROBE holdout=6.38` 那条能力**不存在**

## 40.10 备份位置

`backup-before-rollback-20260912-195936` 分支保留了今天的所有 commit，
以防你想恢复其中非架构的部分（例如 R1b 进程清理教训）。
