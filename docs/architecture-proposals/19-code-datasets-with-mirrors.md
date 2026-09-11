# 19 · 代码能力数据集推荐（含国内镜像站）

> 目标：0.9B 模型、**以代码能力为主**，Chinchilla 比例需 **≈ 18 B token**。
> 代码约 **3.5 byte/token** → **18 B token ≈ 65 GB 原始文本**。
>
> ⚠️ **本文数字来自我的已有知识，未能联网核实**（本会话 `web_search` 报 HTTP 402
> `Insufficient Balance`）。下载前请以页面上显示的实际大小为准。

---

## 〇、镜像站怎么用（先做这一步）

### 方案 A：hf-mirror.com（HuggingFace 国内镜像，最常用）

```bat
:: Windows cmd
set HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"
hf download bigcode/starcoderdata --repo-type dataset --include "python/*" --local-dir D:\data\starcoderdata
```

```bash
# Git Bash / WSL
export HF_ENDPOINT=https://hf-mirror.com
hf download <repo> --repo-type dataset --local-dir <dir>
```

把 `https://huggingface.co/...` 直接换成 `https://hf-mirror.com/...` 即可浏览。

### 方案 B：ModelScope 魔搭（阿里，国内直连最快）

- 网址：**https://www.modelscope.cn/datasets**
- 客户端：`pip install modelscope`
- ```python
  from modelscope.msdatasets import MsDataset
  ds = MsDataset.load('<数据集名>', subset_name='default', split='train')
  ```

### 方案 C：OpenDataLab（上海 AI Lab）

- 网址：**https://opendatalab.com/**
- 万卷（WanJuan）系列官方托管地，国内直连。

### 方案 D：Gitee AI

- 网址：**https://ai.gitee.com/**

> **重要提醒**：`hf-mirror.com` 镜像的是**公开**内容。**gated（需同意条款）**的数据集
> 在镜像上通常**拿不到**，需要 HF 账号 + `HF_TOKEN` 直连。下面凡标 🔒 的都属于此类。

---

## 一、首选：开箱即用的代码预训练语料

| # | 数据集 | 镜像链接 | 规模 | 语言 | 格式 | 备注 |
|---|---|---|---|---|---|---|
| 1 | **StarCoderData** 🔒 | https://hf-mirror.com/datasets/bigcode/starcoderdata | **783 GB / ≈250 B token** | 86 种 | `jsonl.zst` | **最推荐**。已去重 + 已按 opt-out 过滤 + 已质量筛选，BigCode 官方出品，StarCoder 系列原样训练语料 |
| 2 | **python-edu** | https://hf-mirror.com/datasets/HuggingFaceTB/python-edu | **≈55 B token** | Python | `parquet` | **Python 专项质量最高**。教育性过滤（类似 FineWeb-Edu 思路），StarCoder2 用过。**格式匹配仓库现有 parquet 读取器** |
| 3 | **The Stack v2** 🔒 | https://hf-mirror.com/datasets/bigcode/the-stack-v2 | **≈67.5 TB 原始** | 619 种 | 需 S3 | 最大最全，但**需签协议 + 从 S3 拉**，不适合快速起步 |
| 4 | **The Stack (v1) dedup** 🔒 | https://hf-mirror.com/datasets/bigcode/the-stack-dedup | **≈3 TB** | 300+ 种 | `parquet` | 经典。需自己再做质量过滤 |
| 5 | **CodeParrot-clean** | https://hf-mirror.com/datasets/codeparrot/codeparrot-clean | **≈50 GB** | Python | `parquet` | 小、干净、**适合先跑通管线** |
| 6 | **The Stack smol** | https://hf-mirror.com/datasets/bigcode/the-stack-smol | **≈1 GB** | 多语言 | `parquet` | 冒烟测试专用 |
| 7 | **OpenCoder annealing** | https://hf-mirror.com/datasets/OpenCoder-LLM/opc-annealing-corpus | 中等 | 代码+网页 | `parquet` | 高质量退火语料，适合训练末期 |

---

## 二、备选（含代码子集的通用语料）

| 数据集 | 镜像链接 | 代码占比 | 说明 |
|---|---|---|---|
| **Dolma** | https://hf-mirror.com/datasets/allenai/dolma | 有 `starcoder` 子集 | 3 T token，完全开源可复现，含数据配方 |
| **RedPajama-1T** | https://hf-mirror.com/datasets/togethercomputer/RedPajama-Data-1T | GitHub 子集约 5 % | 经典复现基线 |
| **The Pile** | https://hf-mirror.com/datasets/EleutherAI/the-pile | GitHub 子集约 7 % | 多样性好 |
| **WanJuan-1.0** | https://opendatalab.com/ | 含代码子集 | **中文**多来源，Apache-2.0 |
| **FineWeb-Edu** | https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu | 0 %（自然语言） | **配比用**，见第四节 |

---

## 三、训练配比：**不要 100 % 纯代码**

纯代码训练会**显著损害自然语言能力**（CodeLlama / StarCoder 系列都掺了自然语言）。

| 目标 | 代码 | 自然语言 | 说明 |
|---|---|---|---|
| **纯代码能力优先**（你的要求） | **85 %** | 15 % | 保住基本对话/指令理解 |
| 代码+通用均衡 | 50 % | 50 % | 通用模型 |
| 代码专精（不推荐） | 100 % | 0 % | 自然语言会退化 |

**15 % 自然语言用 `FineWeb-Edu`**（https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu），
取 `sample/10BT/` 子集即可。

---

## 四、给你的一条具体配方（18 B token，代码为主）

```
代码 15.3 B token (85%)
  ├─ StarCoderData 精选语言      ≈ 12 B token
  │    python / javascript / java / cpp / c / typescript /
  │    php / go / rust / ruby / csharp / sql / shell
  └─ python-edu                  ≈ 3.3 B token   (Python 质量拔高)
自然语言 2.7 B token (15%)
  └─ FineWeb-Edu sample/10BT     ≈ 2.7 B token
```

**磁盘需求**：原始 ≈ 65 GB + 解压/中间产物 → **预留 200 GB**。
（D: 盘余量我无法实测，`wmic` 在本环境不可用，请你确认。）

---

## 五、最小起步路径（建议顺序）

```bat
:: 第 0 步：先冒烟，别一上来拉 783 GB
hf download bigcode/the-stack-smol --repo-type dataset --local-dir D:\data\smol
hf download codeparrot/codeparrot-clean --repo-type dataset --local-dir D:\data\pyclean

:: 第 1 步：跑通 导出 -> 训练 全链路（小语料）
:: 第 2 步：确认无误后，再拉正式语料
hf download HuggingFaceTB/python-edu --repo-type dataset --local-dir D:\data\python-edu
hf download bigcode/starcoderdata --repo-type dataset --include "python/*" "javascript/*" "java/*" "cpp/*" --local-dir D:\data\scd
```

---

## 六、必须同时做的三件事（否则语料用不上）

| # | 事项 | 原因 |
|---|---|---|
| 1 | **重训 BPE 词表** | 现有 `formal_tokenizer.bbp` 仅 **16 384** 词表且只覆盖合成语料。代码场景建议 **49 152 / 65 536**。已加 `TAO_ALLOW_TOKENIZER=1` 放行冻结校验 |
| 2 | **确认导出器能吃这些格式** | 仓库 `export_repair_bpe_train.cpp` 读 **parquet**。**StarCoderData 是 `jsonl.zst`，需要新增读取分支**；python-edu / CodeParrot / The Stack 是 parquet，可直接用 |
| 3 | **流式分片** | 18 B token 无法一次读入内存。`train_noffn_probe.cu` 现为「整文件读入」，需改成按 shard 流式 |

---

## 七、待你确认

1. **HF 账号能否用**（决定能不能拿 🔒 的 StarCoderData）？
2. **主语言**：Python 优先，还是多语言？
3. **D: 盘余量**是否 ≥ 200 GB？
4. 能否直连 `hf-mirror.com`（公司网络有时会拦）？
