# Engram-Rule 离线解码器

**基线（2026-10-12 换版）**：d=128 / n_layers=2 / n_heads=8 / n_slots=4096 /
**N=2,952 训练行 / 462 步（epochs 40.065）/ seed=11** /
**1,647,150 可训练参数（state_dict 1,651,258）** / 6.34 MB 权重

上一版基线是 d=128 / 400 步 / N=738（§64.8）。**架构未变**（cfg 逐字同构），
所以参数量与上一版完全相同；变的是训练数据量（738 → 2,952）与步数（400 → 462）。

## 这个包是什么

一个**自包含**的推理包：把基线权重 + tokenizer + 解码器打在一起，
**不需要** `engram_rule` 仓库的任何其他文件，也不需要 `train.py`。

生成走的是**出货的** `greedy_generate`（`dialogue_small.py` 里那份），
不是重写的循环 —— 这份解码器已被 §51 的 18 项等价性判据（`check_fast_path_equiv.py`）看守。

## 依赖

| 包 | 用途 | 必须 |
|---|---|---|
| `torch` | 推理 | **是** |
| `tokenizers` | 载入 BPE tokenizer | **是** |

不需要 `requests`（那只用于下载语料）、不需要 `train.py`（那只用于训练与存档）——
精简副本里已去掉这两处 import，其余代码**逐字未改**。

## 用法

```bash
# 交互模式(每行一个问题, 空行退出)
python decode.py

# 单次提问
python decode.py "你最喜欢的食物是什么？"

# 一次多问(自动走等长分组批生成)
python decode.py "你最喜欢的食物是什么？" "今天天气如何？"

# 从文件批量(一行一问)
python decode.py --file questions.txt

# 只打印权重信息(含参数量)
python decode.py --info

# 速度基准(64 行批生成)
python decode.py --bench

# 指定权重 / 设备
python decode.py --ckpt path/to/other.pt --device cpu "问题"
```

## 本基线的实测表现

抽样 200/侧、seed=12345、greedy、max_new_tokens=140、context_limit=512
（`QA_REPORT_n2952d128s462.md`）：

| 侧 | EM | EM 率 | char-F1 | 说明 |
|---|---:|---:|---:|---|
| train（训练见过的 2,952 行内抽样） | 176/200 | **88.00%** | 0.9551 | 只证明记住了 |
| held（完全没见过） | 0/200 | **0.00%** | 0.1194 | **泛化仍然为零** |

held 侧**教师强制** token accuracy = **0.297571**（answer_nll 6.312596 / ppl 551.47）。
⚠️ **tok_acc 与 EM 是两个口径**：tok_acc 只要求逐 token 落在正确答案位置，
EM 要求整句全对且解码走对路径。前者的提升**不会**自动变成后者。

### ★两个必须知道的边界

1. **这是记忆基线，不是泛化基线。** held EM = 0/200；held 的输出多为
   **通顺但无关的中文**（EOS 率 100%、截断率 0%）⇒ 解码器正常，是记忆/泛化问题。
2. **身份问句无法回答。** 当前语料（`corpus_modelscope_v2_155740.jsonl`，155,740 行）
   里 **身份问句 0 条**（「你叫」/「你是谁」/「你的名字」/「自我介绍」在 question 字段各 0 条），
   连名字本身也 0 次。实测 `你的名字是什么` → `鼓浪屿是厦门旗下`。
   ⇒ 模型不是「学不好身份」，是**从未被教过**；这与 N、d、步数都无关。

## 解码参数

- `greedy`（`temperature=0.0`）
- `max_new_tokens = 400`（§60.5 口径：语料 gold_len 最大 366）
- `context_limit = 512`（cfg 的 `pos_abs_max`）
- `utf8_safe = True`（ByteLevel 允许集，保证不会吐出半个汉字）
- `repetition_penalty = 1.0`、`no_repeat_ngram = 0`（出货默认，未加任何约束）

## 文件清单

| 文件 | 大小 | 说明 |
|---|---:|---|
| `engram_rule_nsweep_n2952_d128_s462_wd0p0.pt` | 6.34 MB | 权重 + cfg + tokenizer + cell |
| `model.py` | 0.041 MB | 模型定义，**逐字拷贝** |
| `dialogue_small.py` | 0.090 MB | 解码器，含 §69.23 的 think 支持 |
| `decode.py` | 0.005 MB | CLI 入口 |
| `README.md` | — | 本文件 |

## ⚠️ 不要用 `_pkg1.py` 重新生成本目录的模块副本

`dist/dialogue_small.py` **不是**主目录 `dialogue_small.py` 的机械副本：
§69.23 的 think 支持**只加在 dist 这一份**上（主目录那份停在 2026-10-10 的旧版）。
实测重新生成会让 `answer_loss_sums` 退回无 think 参数的旧实现
（diff 见 §69.52.9），这会改掉 MGATE 指纹并打断
`check_think_backcompat.py` 的 B1/B2。

⇒ **换基线只需换权重 + 改 `decode.py` 的 `DEFAULT_CKPT` + 更新本文件**，
模块副本不要动。

## 已知边界

- 单 seed（seed=11），**无种子极差**；每侧只抽 200 条，held 的 EM 分辨率是 1/200。
- 权重里**不含** `raw_dialogue_rows`。
- 上一版基线权重 `engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st400.pt` 已从本目录移除。
