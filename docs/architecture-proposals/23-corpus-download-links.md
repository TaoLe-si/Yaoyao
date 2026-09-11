# 23 · 长训语料清单与下载链接

> **验证状态说明**：本文档编写时 `web_search` 不可用（HTTP 402 余额不足），
> 所有链接、体积、许可来自模型知识，**未经实时核对**。
> 每一条都给出了可直接执行的验证命令，请在下载前先跑一遍。
> 已知可用的镜像：`https://hf-mirror.com`（实测连通）。

---

## 零、先确定要多少 token

架构固定：**H2R，2 层，d=512，s=128，m=512，e=1024，vocab=16384**。
当前参数量约 **4.5 M**（这是探针规模，不是 0.9B）。

| 目标规模 | 参数量 | Chinchilla 最优 token | 实际建议 |
|---|---:|---:|---:|
| 当前探针 | ~4.5 M | 0.09 B | 0.3–1 B |
| 中间验证 | ~30 M | 0.6 B | 2–3 B |
| **0.9B 正式** | ~0.90 B | 18 B | **20–30 B**（含代码重复采样） |

**关键**：在架构缺陷（doc 22：顺序/计数/集合容量）修好之前，
**不要开始 0.9B 长训**。否则只是把缺陷放大 200 倍。

---

## 一、代码语料（主力，用户明确要求"主要是代码能力"）

### 1.1 `codeparrot/github-code` —— **首选，无 gating**

| 项 | 值 |
|---|---|
| 链接 | https://huggingface.co/datasets/codeparrot/github-code |
| 体积 | ~1 TB（全量，115 M 文件） |
| 语言 | 32 种 |
| 许可 | 仅 permissive（MIT / Apache-2.0 / BSD / ISC 等），已过滤 |
| gating | **无**，可直接下载 |
| 形态 | parquet |

**推荐子集**（只下需要的语言，体积可控）：

```bash
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com          # Linux/macOS
set HF_ENDPOINT=https://hf-mirror.com             # Windows cmd

# 只下 Python + C++ + JS + TS + Rust + Go + Java（按需增减）
hf download codeparrot/github-code --repo-type dataset \
  --include "data/python/*" "data/c++/*" "data/javascript/*" \
            "data/typescript/*" "data/rust/*" "data/go/*" "data/java/*" \
  --local-dir E:/corpus/github-code
```

### 1.2 `bigcode/starcoderdata` —— 质量最好，**需要 gating**

| 项 | 值 |
|---|---|
| 链接 | https://huggingface.co/datasets/bigcode/starcoderdata |
| 体积 | ~783 GB |
| 语言 | 86 种 |
| 许可 | The Stack 的 permissive 子集 + GitHub issues |
| gating | **需要登录 HF 并接受条款** |
| 形态 | parquet |

```bash
hf auth login          # 需要 token
# 先在网页上点 Accept：https://huggingface.co/datasets/bigcode/starcoderdata
hf download bigcode/starcoderdata --repo-type dataset \
  --include "python/*" "c++/*" "javascript/*" \
  --local-dir E:/corpus/starcoderdata
```

> **若无法通过 gating**：用 `codeparrot/github-code`（1.1）替代，许可更宽松且无门槛。

### 1.3 `bigcode/the-stack-smol` —— **小样本，先跑通流程用**

| 项 | 值 |
|---|---|
| 链接 | https://huggingface.co/datasets/bigcode/the-stack-smol |
| 体积 | ~1 GB |
| gating | 无 |
| 用途 | **先用它验证整条流水线，再下大语料** |

```bash
hf download bigcode/the-stack-smol --repo-type dataset --local-dir E:/corpus/stack-smol
```

### 1.4 `codeparrot/codeparrot-clean` —— 纯 Python，已清洗

| 项 | 值 |
|---|---|
| 链接 | https://huggingface.co/datasets/codeparrot/codeparrot-clean |
| 体积 | ~22 GB |
| 语言 | Python |
| gating | 无 |

```bash
hf download codeparrot/codeparrot-clean --repo-type dataset --local-dir E:/corpus/codeparrot-clean
```

### 1.5 指令/对话代码数据（小，用于 SFT 式微调）

| 数据集 | 链接 | 体积 |
|---|---|---|
| `HuggingFaceH4/CodeAlpaca_20K` | https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K | ~20 MB |
| `sahil2801/CodeAlpaca-20k` | https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k | ~20 MB |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k | ~50 MB |
| `microsoft/CodeXGLUE` | https://huggingface.co/datasets/microsoft/CodeXGLUE | ~几 GB |

---

## 二、自然语言语料（辅料，保持语言能力）

### 2.1 `HuggingFaceFW/fineweb-edu` —— **首选英文**

| 项 | 值 |
|---|---|
| 链接 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| 体积 | 全量 ~2.5 TB / 1.3 T token；`sample-10BT` 子集 ~30 GB |
| 许可 | ODC-By 1.0 |
| gating | 无 |

```bash
# 建议只下 10BT 或 100BT 子集
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
  --include "sample/10BT/*" --local-dir E:/corpus/fineweb-edu-10bt
```

### 2.2 `HuggingFaceFW/fineweb` —— 未过滤版，更大

https://huggingface.co/datasets/HuggingFaceFW/fineweb （~44 TB，`sample-10BT` ~30 GB）

### 2.3 `allenai/c4` —— 经典，稳妥

| 项 | 值 |
|---|---|
| 链接 | https://huggingface.co/datasets/allenai/c4 |
| 体积 | en ~750 GB；`en.noclean` 更大 |
| gating | 无 |

### 2.4 中文语料

| 数据集 | 链接 | 说明 |
|---|---|---|
| `opencsg/chinese-fineweb-edu` | https://huggingface.co/datasets/opencsg/chinese-fineweb-edu | 中文高质量，推荐 |
| `liwu/MNBVC` | https://huggingface.co/datasets/liwu/MNBVC | 超大中文，需挑子集 |
| `BAAI/IndustryCorpus2` | https://huggingface.co/datasets/BAAI/IndustryCorpus2 | 行业中文 |
| `wikipedia` (zh) | https://huggingface.co/datasets/wikimedia/wikipedia | `20231101.zh` |

### 2.5 数学（补计算能力，对应 doc 22 缺陷 2）

| 数据集 | 链接 | 体积 |
|---|---|---|
| `HuggingFaceTB/finemath` | https://huggingface.co/datasets/HuggingFaceTB/finemath | ~几百 GB |
| `openai/gsm8k` | https://huggingface.co/datasets/openai/gsm8k | ~几 MB（评测） |
| `deepmind/math_dataset` | https://huggingface.co/datasets/deepmind/math_dataset | ~几 GB |

---

## 三、镜像与加速

```bash
# 全局镜像（已验证可用）
export HF_ENDPOINT=https://hf-mirror.com

# 国内加速下载
pip install -U hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**注意**：`HF_ENDPOINT` 必须在下载**之前**设置；已缓存的元数据不会重新走镜像。

---

## 四、磁盘与落盘规划

| 盘 | 可用 | 建议用途 |
|---|---:|---|
| C: | 44 GB | 不放语料 |
| D: | 431 GB | 代码 + 工具链 |
| **E:** | **626 GB** | **语料主库** |

推荐目录：

```
E:/corpus/
  github-code/          # 主力代码（按语言分子目录）
  starcoderdata/        # 若有 gating 权限
  fineweb-edu-10bt/     # 英文
  chinese-fineweb-edu/  # 中文
  finemath/             # 数学
E:/corpus/sharded/      # 经 build_corpus 处理后的 TLP3 分片
```

---

## 五、下载前先跑这 4 条验证（因为本文未实时核对）

```bash
# 1) 数据集是否存在
hf download bigcode/the-stack-smol --repo-type dataset --local-dir /tmp/probe

# 2) 是否 gating
python -c "from huggingface_hub import dataset_info; print(dataset_info('bigcode/starcoderdata').gated)"

# 3) 真实体积
python -c "from huggingface_hub import list_repo_files; print(len(list_repo_files('codeparrot/github-code',repo_type='dataset')))"

# 4) 镜像连通
curl -sI https://hf-mirror.com | head -1
```

---

## 六、语料配比建议（代码为主）

按用户要求"主要是代码能力"，且当前架构缺陷集中在**顺序/计数/计算**：

| 类别 | 占比 | 理由 |
|---|---:|---|
| 代码（多语言） | **60%** | 主力能力 |
| 中文自然语言 | 20% | 对话与语言基础 |
| 英文自然语言 | 10% | 代码注释/文档 |
| 数学/推理 | 10% | **直接针对 doc 22 缺陷 2（计算）** |

**课程三段映射**（doc 20）：

| 阶段 | 内容 | 判据 |
|---|---|---|
| 精 core | 最小最干净：`the-stack-smol` + `CodeAlpaca_20K` + `fineweb-edu` 精选 | 验证 NLL 饱和 |
| 混合 mixed | 逐步提高代码比例，加入中文 | NLL 不反弹 |
| 干扰 interfere | 同姓不同名、共享前缀问题、近似 API | **专治查表行为** |

---

## 七、尚未解决的前置问题

| # | 问题 | 状态 |
|---|---|---|
| 1 | 架构顺序/计数缺陷（doc 22） | **阻塞长训** |
| 2 | 词表 16384 → 49152/65536（代码需要更大词表） | 待办 |
| 3 | 训练器文档长度上限 512（代码文档通常更长） | 待办 |
| 4 | 分片流式读取接入训练器（当前整份读入内存） | 待办 |
| 5 | `build_corpus` / `shard_corpus` 端到端验证 | 待办 |
| 6 | 语料下载后需转成 TLP3（`plain_lm_format.hpp`） | 已实现 |
