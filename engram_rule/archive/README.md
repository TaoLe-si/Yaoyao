# archive/ — 历史文件

**归档时间**：2026-10-10。**归档原因**：桃「以 400 步那个为基准，其余的做归档处理，保持整洁」。

**归档 ≠ 删除。** AGENTS.md 规则 15/16：被否证的假设与历史读数必须留在文档里。
这里每个文件都还在，只是不在主目录占位置。要找什么，先看下面的索引。

## 基准（不在本目录，在上级）

| 类别 | 文件 |
|---|---|
| 权重 | `../engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st400.pt`（6.644 MB / 1,651,258 参数） |
| 语料 | `../dialogue_qa_answer_disjoint_v11.jsonl` |
| 设计 | `../DESIGN.md`（§64.8 是基准定档） |
| 报告 | `../train_qa_report_d128_st400.md` |
| 训练器 | `../exp_train_budget.py` + `../dialogue_small.py` + `../model.py` |
| 读数 | `../train_budget_v1.jsonl`（400 步档）、`../scaling_grid_v1.jsonl` |

## 按 DESIGN 章节索引

### 权重（`.pt`）

| 文件 | 大小 | 说明 / 对应章节 |
|---|---:|---|
| `engram_rule_dialogue_bpe_answer_disjoint_v11.pt` | 3.995 MB | **d=64 出厂权重**（767K 参数）。§38/§59/§61 全部早期读数的锚点，**不能删**。 |
| `engram_rule_dialogue_bpe_answer_disjoint_d128_s11.pt` | 6.556 MB | §59 的 d=128 / 200 步（N=820）。被 st400 取代；**它含 `raw_dialogue_rows`**，st400 版本不含。 |
| `engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st200.pt` | 6.336 MB | §64 训练预算扫描 200 步档。 |
| `engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st800.pt` | 6.336 MB | §64 的 800 步档（已饱和）。 |
| `engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st1600.pt` | 6.336 MB | §64 的 1600 步档（过拟合，探针损失最高 8.4631）。 |
| `engram_rule_dialogue_qpool_s11.pt` | 2.994 MB | §62 问题池化 arm（pool）。**判决未命中**，存档备查。 |
| `engram_rule_dialogue_bpe_answer_disjoint_v10.pt` | 3.974 MB | §49 之前的中间版本。 |
| `engram_rule_dialogue_bpe_answer_disjoint_smoke_v{5..9}.pt` | 各 2.948 MB | 冒烟测试权重。 |
| `engram_rule_dialogue_bpe_answer_disjoint.pt` / `_v2.pt` / `_bpe_small.pt` | 各 2.676 MB | 最早的 BPE 实验。 |
| `engram_rule_dialogue_bpe_learning_v3.pt` / `_answer_disjoint_learning_v3.pt` | 各 2.687 MB | 学习率实验。 |
| `engram_rule_dialogue_bpe_answer_disjoint_weight{1,8}_learning_v4.pt` | 各 2.779 MB | `first_token_weight` 消融（§36 附近）。 |
| `engram_rule_dialogue_memorize8_*.pt` | 各 ~2.68 MB | 「记 8 条」系列实验（早于 answer-disjoint）。 |

### 读数（`.jsonl`）

| 文件 | 说明 / 对应章节 |
|---|---|
| `free_generation_v1.jsonl` | §56 路径一致性判决 + d=64/128 单 seed 逐行读数。**含多期 gate，旧 gate 已被 §56 判为作废**——只有 gate 与当时脚本一致的那条有效。 |
| `qpool_v1.jsonl` | §62 池化 A/B 两条 arm 的逐行读数（gate `a62184abc3f6`）。 |
| `capacity_scaling_v1.jsonl` | §48 容量实验 13 条 run 记录（4 precheck + 9 main）。 |
| `dialogue_qa_answer_disjoint_v11_train_fit_review.jsonl` | §46 的 train 拟合复查。 |
| `dialogue_qa_answer_disjoint_v{10,...}.jsonl` / `_smoke_v*` / `_weight{1,8}` / `_learning_v*` | 各代语料与中间产物。 |
| `dialogue_qa_memorize8_*.jsonl` | 「记 8 条」系列读数。 |

### 脚本（`.py`）按用途

| 组 | 文件 | 对应章节 |
|---|---|---|
| **判决** | `probe_free_generation.py`、`probe_scaling_fit.py`(已留)、`probe_capacity_scaling.py` | §56 / §48 |
| **测量** | `probe_rerank_mbr.py`、`probe_question_conditionality.py`、`probe_delta_conditioning.py`、`probe_window_question_reach.py`、`probe_long_sentence_degrade.py`、`probe_context_capacity.py`、`probe_window_coverage.py`、`probe_opener_learnability.py`、`probe_role_vs_term.py`、`probe_generalization_decomp.py`、`probe_length_confound.py`、`probe_first_token_ablation.py`、`probe_engram_slots.py`、`probe_eval_baseline.py`、`probe_train_subset_anchor.py`、`probe_train_qa_report.py`、`probe_train_mem_growth.py`、`probe_padding_waste.py` | §50 / §55 / §42 / §61 / §60 / §53 / §41 / §46 / §45 / §44 / §40 / §47 / §49 |
| **实验** | `exp_capacity_scaling.py`、`exp_free_generation.py`、`exp_qpool.py`、`exp_window_move.py`、`exp_agreement*.py`、`exp_held_nouns.py`、`exp_replace.py`、`memorize_dialogue_8.py`、`eval_solid.py`、`small_corpus.py`、`corpus_agreement.py`、`train.py` | §48 / §56 / §62 / §39 |
| **性能** | `profile_generation_path.py`、`profile_train_step.py`、`profile_training_speed.py`、`bench_train_trim.py`、`bench_gpu.py`、`bench_crossover_recheck.py` | §51 / §52 / §43 / §13 |
| **门禁** | `check_free_generation_probe.py`、`check_bucket_resolution.py`、`check_prompt_conditioning.py`、`check_generation_context.py`、`check_memorize_dialogue_8.py`、`audit_v11_config.py`、`audit_impl.py` | 各节 |
| **诊断** | `diag_bimodal.py`、`diag_det.py`、`diag_emit.py`、`diag_gate.py`、`diag_reconcile.py`、`verify_copy_leak.py`、`verify_held100.py` | §16–§24 系列 |

### 日志（`.log`）

全部是历次跑批的控制台重定向，保留用于回溯某次运行的实际输出。

### 报告（`.md`）

| 文件 | 说明 |
|---|---|
| `train_qa_report.md` | v11（d=64）出厂模型的 12 条自由生成样本 —— **「答非所问」的原始证据**，§58 的起点。 |
| `train_qa_report_d128.md` | §59 的 d=128 / 820 行全表（200 步口径，**早于** §64 的 400 步）。 |
| `SPEC_v1.1.md` / `SPEC_v1.2.md` | 早期规范；现用 `../SPEC_v1.4.md`。 |

## 重要提醒

1. **`engram_rule_dialogue_bpe_answer_disjoint_v11.pt` 是 d=64 的锚点**——所有早期章节
   （§38/§42/§46/§49/§55/§61）的读数都基于它。要复现那些读数，从它加载。
2. **`free_generation_v1.jsonl` 里有多个 gate**——每改一次测量脚本，gate 就变。
   只有 `gate` 与当时的 `exp_free_generation.py` + 检查脚本一致的那条才有效，
   其余是 §56.4 记录的「改完没重跑检查」的产物，**已被门禁判为作废**。
3. **`_d128_s11.pt` 与 `_d128_s11_st400.pt` 内容不同**：前者含 `raw_dialogue_rows`
   （下游脚本可能需要），后者不含但步数是 400。
