# 41 · 今天设计的完整训练流程

> **引用规则**：所有行号均以备份分支 `backup-before-rollback-20260912-195936`
> 中的源文件为准（`git show backup-before-rollback-20260912-195936:<path>`）。
> HEAD 不再指向这份代码 —— 它已被回退。

## 41.1 与回滚版本（7393d2c）的整体差异

| 维度 | 回滚版本（7393d2c） | 今天的设计（backup-before-rollback-20260912-195936） |
|---|---|---|
| 模型形状 | d=512, s=128, m=512, e=1024, dk=64 | **d=3200, s=1600, m=3200, e=??, dk=400**（GPT-1 同级） |
| 参数量 | ~10M | **119,033,986** |
| Width 下界 | `graph_width >= 8` | `graph_width >= 1`（用 width=1 换更大 GEMM batch） |
| 续训 | 必须换 `OUT_DIR` | **原地续训允许**（仅当 RESUME_DIR == OUT_DIR） |
| 收敛判据 | 无 | **每片训到收敛才进下一片**（TAO_CONVERGE=1） |
| 留出集 | 无 | 每片末尾 128 篇不参与训练，每 150 步评一次留出 NLL |
| 过拟合下界 | 无 | 训练 NLL < TAO_CONV_FLOOR 立即停 |
| 学习率 | 1e-3 恒定，warmup 硬编码 20 | 1e-3 默认，**TAO_LR 可覆盖**，**warmup 可配** |
| 阶段计时 | 无 | `TAO_STAGE_TIMING=1` 输出每步 ms_graph/ms_upd/... |
| 生成 | 只有续写 | `build/gen_text.exe` 纯文本续写 + `build/grpo_rollout.exe` 问答采样 |
| 评测 | `eval_nll` 裸文本 | `eval_nll --frame 1` 复刻 TLP3 边框 + `--wrap 1` |
| 量化 | fp32 落盘 | **2-bit 打包** `final.dsb`，常驻 CPU，AVX2/VNNI 内核 |
| m-phase | 朴素 reduction | **显式 `_mm256_fmadd_ps` 向量化**（3a302cd） |
| 重复惩罚 | 无 | 解码端 multiplicative repetition penalty |

## 41.2 顶层流水线

```
原始语料 ─build_corpus─→ TLP2/TLP3 分片 ─train_shards─→ pretrained .dsb
                                                            │
                                                            ▼
                                          ┌── gen_text.exe   纯文本续写
                                          ├── eval_nll.exe   留出 NLL
                                          └── grpo_rollout   问答采样
```

**整个流水线靠环境变量注入全部配置**（设计原则 —— 不是 CLI 参数）：

| 变量 | 作用 | 默认 |
|---|---|---|
| `TAO_ALLOW_TOKENIZER` | 跳过冻结分词器 digest 校验 | 未设=校验 |
| `TAO_TOKENIZER` | 分词器路径 | (cli argv[2]) |
| `TAO_CFG_LAYERS` | 架构层数 | 2 |
| `TAO_CFG_D` | 主维度 d | 512 |
| `TAO_CFG_S` | 状态维度 s | 128 |
| `TAO_CFG_M` | 记忆维度 m | 512 |
| `TAO_CFG_E` | embedding（未显式使用） | 1024 |
| `TAO_CFG_DK` | key/query 维度 | 64 |
| `TAO_CFG_VOCAB` | 词表 | 16384 |
| `TAO_OPT_STATE` | 是否写 opt_state.bin | 1 |
| `TAO_OPT_OFFLOAD` | Adam 状态驻 host RAM | 0 |
| `TAO_LR` | 学习率 | 1e-3 |
| `TAO_WARMUP` | warmup 步数 | 20 |
| `TAO_SHUFFLE_SEED` | 分片洗牌种子基值 | 20260912 |
| `TAO_HOLDOUT` | 每片末尾留出篇数 | 0（无留出） |
| `TAO_HOLDOUT_TOKENS` | 每篇留出最多评 token 数 | 64 |
| `TAO_CONVERGE` | 是否启用收敛门控 | 未设=关 |
| `TAO_CONV_WINDOW` | 收敛判据的滑窗步数 | 200 |
| `TAO_CONV_Z` | 收敛判据 Z 阈值 | 2.0 |
| `TAO_CONV_TOL` | 收敛判据相对容差 | 0.001 |
| `TAO_CONV_PATIENCE` | 连续低收益窗口数 → 收敛 | 3 |
| `TAO_CONV_MIN` | 收敛最小步数 | 200 |
| `TAO_CONV_MAX` | 收敛最大步数 | =steps_per_shard |
| `TAO_CONV_FLOOR` | 过拟合下界 | 0.2 |
| `TAO_CPU_THREADS` | CPU 留出评/推理线程数 | 8 |
| `TAO_STAGE_TIMING` | 启用阶段计时 | 0 |
| `TAO_FAST_ACT` | 快速 Padé 激活 | 1 |

## 41.3 训练入口：`build/train_shards.exe`

### 命令行
同回滚版（备份 `src/train_shards.cu` line 48-57）：

```
train_shards SHARD_DIR TOKENIZER.bbp OUT_DIR STEPS_PER_SHARD
            [SLOTS WIDTH] [RESUME_DIR] [START_SHARD]
```

### 形状
**今天**默认形状（通过环境变量注入，不是默认 Config）：

```bash
export TAO_CFG_LAYERS=2 TAO_CFG_D=3200 TAO_CFG_S=1600 TAO_CFG_M=3200 TAO_CFG_DK=400
```

参数量 119,033,986（≈117M GPT-1）。

### 训练循环（备份 `train_shards.cu:222-310`）
```cpp
for si in start_shard..shards.size():
  docs = read_shard()
  hold_docs = last hold_n docs     // 留出切分
  docs.resize(docs.size() - hold_n)
  PilotCursor(docs, slot_count, false)
  ShuffledEpochCursor(..., seed_base+si, identity)
  if TAO_CONVERGE:
    while u < conv_max and (not converged):
      do_step()
      if u % conv_window == 0: maybe_converged()   // CONV_PROBE 输出
  else:
    for u in 0..steps_per_shard: do_step()
```

### `do_step`（备份 `train_shards.cu:264-310`）
1. 8 个 round × `replay.run(p, slots)`
2. `lr = steps<warmup ? lr_max*(steps+1)/warmup : lr`
3. `tr.update(targets, lr)`
4. `if tr.steps%50==0: export_ckpt(false)`
5. 如果 `TAO_CONV_WINDOW` 步过：评估 `holdout NLL`，输出 `CONV_PROBE`

## 41.4 收敛门控（核心新设计）

备份 `train_shards.cu:184-219`。

**判定**：

每 `TAO_CONV_WINDOW` 步：
1. 累积窗口内 sum 与 sumsq → 标准差 → 均值标准误 SE
2. 上一窗口均值 - 本窗口均值 > Z × SE **且** 相对改进 > TOL
   → 本窗口"真的还在学"
3. 连续 `TAO_CONV_PATIENCE` 个窗口不满足 → 判定本片收敛
4. 训练 NLL < `TAO_CONV_FLOOR` → 立即停（默认 0.2）

**为什么用 Z 而非固定容差**：噪声实测 ±0.5~2%，固定小容差会被噪声主导。

**过拟合下界实测依据**：0.2 只在 0.2~3MB 闭合合成语料上可达；
581MB 真实开放语料 29,065 步里从未低于 0.88，故该下界在真实语料上不会误触发。

## 41.5 留出集评估

备份 `train_shards.cu:228-232, 250-289`。

- **切分**：每片**末尾** `TAO_HOLDOUT` 篇（如 128）不参与训练
- **每篇只评前** `TAO_HOLDOUT_TOKENS` 个**有监督** token（默认 64）
- **逐文档计 NLL**，配对差检验（窗口间的差），方差远小于独立均值
- **拷权重到 CPU**（`tr.graph.w.at(...)->value.host()`）→ `CpuModel` →
  `GreedyPipelineGroupedModel`（**注意**：用的是 `GreedyPipelineGroupedModel` 的
  推理路径，不是 trainer 内部的图）
- 多线程：`TAO_CPU_THREADS`（默认 8）

## 41.6 原地续训守卫（修复）

备份 `train_shards.cu:81-86`。

```cpp
const bool resume_in_place = !resume_dir.empty() &&
    fs::equivalent(fs::path(resume_dir), out, std::error_code{});
require(!out.empty() && (resume_in_place || !fs::exists(fs::symlink_status(out))),
        "output directory exists; never overwrite (pass RESUME_DIR=OUT_DIR to resume in place)");
```

- `RESUME_DIR == OUT_DIR` → 允许原地续训
- 其他情况仍拒绝覆盖（防止误把无关目录当输出目录）

## 41.7 Stage Timing

备份 `train_shards.cu:118-122, 311-330`。

`TAO_STAGE_TIMING=1` 时每步额外同步两次，输出：

```
ms_graph=... ms_upd=... ms_project=... ms_graph_sum=... ms_upd_sum=... ms_total=...
```

用于分离 CUDA Graph replay 与 Adam 更新，找瓶颈。

## 41.8 生成：`build/gen_text.exe`

备份源码 `src/gen_text.cpp`（**新增文件**）。

- 输入 `--model final.dsb`、`--prompt-file foo.txt`、`--max N`
- **只注入 BOS，不注入任何角色标记**
- 反复 `hm.step(prev_id, state)` 贪心解码 N 步
- 输出：`GEN_STEPS distinct=... max_run=... GEN_ENTROPY first=... min=... last=... GEN_TEXT<<<...>>>`
- 这是用来**诊断熵是否均匀**的，不带 role-special

## 41.9 Q&A 采样：`build/grpo_rollout.exe`

备份源码 `src/grpo_rollout.cpp`（**新增文件**）。

- 加载 `final.dsb` 与 `data/grpo_test.jsonl`
- 多次采样（`--samples 8`）每题 64 token
- 3 个奖励：correct / format（含"答案："）/ calc（含"计算"）
- `--prefix train`：注入 `U <q> A ` 前缀（256/257/258/259 角色 token）
- `--prefix raw`：裸 prompt
- 输出 pass@1 / pass@G / 组内方差 / 格式率 / 步骤率
- **GRPO 信号判据**：组内方差 = 0 → 优势无定义 → 学不到东西

## 41.10 评测：`build/eval_nll.exe`

备份源码 `src/eval_nll.cpp`。

CLI 新增：
- `--frame 1`：用 BOS + 正文 + TURN_END 包裹（复刻 TLP3 训练语义）
- `--frame 0`：裸文本（默认）
- `--wrap 1`：`DOC\nU ` 前缀模式（BPE 边界校验）
- `--max-tokens`：限制每篇最多评 token 数
- 输出新字段 `doc_nll_mean / doc_nll_p50 / doc_nll_p90`

## 41.11 参考下界：`build/entropy_ref.exe`

备份源码 `src/entropy_ref.cpp`（**新增**）。

- 一元 / 二元 unigram/bigram，add-one 平滑
- 输入：前一半拟合、后一半评测
- 用作"不作任何学习"的下界，与模型 NLL 对照

## 41.12 解码端：`build/tlp3_dump.exe`

备份源码 `src/tlp3_dump.cpp`（**新增**）。

- 把 TLP2/TLP3 分片解回明文
- 用于把训练器 holdout docs 翻译成可读文本，便于核对留出集内容

## 41.13 内核改动（性能）

备份 diff 中性能相关：

| commit | 内容 | 实测 |
|---|---|---|
| `ba298a6` | ternary 权重 **2-bit 打包** | decode 387→458 tok/s |
| `0e21f56` | VNNI 输入量化向量化 + 组内去重 | m-phase -23% |
| `3a302cd` | m-phase 显式 AVX2 reduction（`_mm256_fmadd_ps`） | m-phase -28%, total -18% |
| `36e75ae` | parallelize delta kernels + pin optimizer state + VRAM cliff fix | 200 pos/s → 3635 pos/s |
| `16229d5` | restore packed VNNI kernel + fuse delta-memory update | 2636 → 4220 pos/s |

**最终部署的吞吐**：CPU 解码 ~450 tok/s，GPU 训练 ~0.43 steps/s（slots=16, width=3, 384 pos/step）。

## 41.14 完整运行实例

```bash
# 1) 训练（环境变量注入 GPT-1 形状）
export TAO_ALLOW_TOKENIZER=1
export TAO_TOKENIZER=D:/TaoVm/build/tok_v2.bbp
export TAO_CFG_LAYERS=2 TAO_CFG_D=3200 TAO_CFG_S=1600
export TAO_CFG_M=3200 TAO_CFG_DK=400 TAO_OPT_OFFLOAD=1
export TAO_HOLDOUT=128 TAO_HOLDOUT_TOKENS=64
export TAO_LR=0.0006 TAO_WARMUP=200 TAO_SHUFFLE_SEED=20261101
export TAO_CONVERGE=1
export TAO_CONV_WINDOW=150 TAO_CONV_Z=2.0 TAO_CONV_TOL=0.0015
export TAO_CONV_PATIENCE=3 TAO_CONV_MIN=400 TAO_CONV_FLOOR=0.2

# 跑分片 0..29，每片训到收敛
build/train_shards.exe data/p1_final build/tok_v2.bbp build/L1_pretrain 32000 16 3

# 2) 评测留出 NLL（按训练同语义）
build/eval_nll.exe \
  --model build/L1_pretrain/step_3050/final.dsb \
  --text build/holdout_shard0.txt \
  --tokenizer build/tok_v2.bbp --threads 8 --frame 1

# 3) 参考下界（同批次）
build/entropy_ref.exe --text build/fit_then_eval.txt --tokenizer build/tok_v2.bbp

# 4) 纯文本续写
build/gen_text.exe \
  --model build/L1_pretrain/step_3050/final.dsb \
  --prompt-file build/p1.txt \
  --tokenizer build/tok_v2.bbp --max 48

# 5) 问答采样
build/grpo_rollout.exe \
  build/L1_pretrain/step_3050/final.dsb \
  data/grpo_test.jsonl \
  --n 40 --samples 8 --max 64 --show 40 --prefix train --seed 1
```

## 41.15 与回滚版本对比 —— "为什么"

- **模型放大**：d=512 → d=3200 → 接近 GPT-1 117M；
  老的 d=512 没法生成连贯中文，需要 GPT-1 级别的容量
- **width 下界放宽**：旧版 width=8 起对吞吐是死限，
  width=1 才能把 GEMM batch 堆到 slots（更多并发）
- **2-bit 打包**：fp32 119M × 4B = 476MB 落盘太大，
  2-bit 28.7MB 即可常驻 CPU，解码时寄存器内 unpack
- **VNNI/AVX2 内核**：手写 SIMD 比 cuBLAS 在 3200×3200×16 GEMM 上更快
- **m-phase 向量化**：delta rule 的 reduction 是热点，逐项加法用 `_mm256_fmadd_ps`
- **原地续训**：kill -9 之后人工重启是常态，要能直接续训而不是换目录
- **收敛门控**：单片训到收敛再换片，避免"前几片没训完就开始走神"
- **留出集 NLL**：train NLL 单调下降，必然"收敛"，要拿 held-out NLL 才算
- **过拟合下界**：避免在闭合合成语料上学到过拟合
- **GRPO rollout**：在 base 上做 SFT 风格的能力探测
- **entropy_ref**：给"不作任何学习"找一个公平的下界对照

## 41.16 为什么回退

用户决定："新设计的架构有非常严重的问题，从 GitHub 回退昨晚训练的架构"。

具体的"严重问题"**没有写在任何文档里**，回滚本身没有附带 commit message。
判断"严重问题"具体是什么需要看新架构哪部分会让模型在测试集上更差。
已知的、可能与"严重"相关的新增设计：

1. **2-bit 打包 + 解码时寄存器 unpack** —— 量化精度损失在 fp32 训练值上
2. **m-phase AVX2 向量化** —— fp reduction 顺序与朴素版不同，相对误差 ~1.9e-5
3. **weight 2-bit 落盘** —— 训练是 fp32，推理是 2-bit，两者不匹配
4. **width 下界 = 1** —— 短 BPTT 改变了 loss landscape
5. **delayed stage timing** —— 可能干扰 CUDA Graph replay 稳定性
6. **新增 5 个可执行**（gen_text、grpo_rollout、entropy_ref、tlp3_dump、eval_nll --frame）

如果想恢复部分今天的设计而不是全部，请按 commit 列表（`git log --oneline
backup-before-rollback-20260912-195936 ^7393d2c`）挑选 cherry-pick。
