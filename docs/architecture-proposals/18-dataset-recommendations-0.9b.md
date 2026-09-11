# 18 · 0.9B 真实训练的数据集推荐与下载方案

> 给用户下载用。目标：0.9B 参数、Chinchilla 比例 **≈ 18 B token**。
> 现有一切语料（L5 合成 0.7 M token + 仓库内 repair 导出 5.95 M token）**差 4 个数量级**，必须外接。

---

## 一、先决条件（先算，再下）

| 项 | 需求 | 说明 |
|---|---|---|
| **token 数** | **≈ 18 B** | Chinchilla 20× 参数。少于 10 B 会明显欠训 |
| **原始文本体积** | **≈ 60–80 GB** | 英文约 4 B/token；中文约 1.5–2 B/token（UTF-8） |
| **磁盘余量** | **≥ 200 GB** | 含原始 parquet + 解压 + 导出后的 train.bin |
| **词表** | **需重新训练 BPE** | 现有 `formal_tokenizer.bbp` 词表仅 **16 384**、且只覆盖合成语料，**必须换**。建议 **32 768 或 65 536** |

> ⚠️ **磁盘余量我没能实测**（`wmic` 在此环境不可用）。请你先确认 D: 盘剩余空间 ≥ 200 GB 再下载。

---

## 二、首选推荐（按「性价比 / 可得性 / 与本架构匹配度」排序）

### 🥇 方案 1：FineWeb-Edu（英文）——最省事、质量最高

| 项 | 值 |
|---|---|
| HuggingFace | **`HuggingFaceFW/fineweb-edu`** |
| 规模 | **1.3 T token**（取其中 18 B 即可，约 1.4 %） |
| 格式 | **parquet**（正好匹配仓库里 `export_repair_bpe_train.cpp` 的读取器） |
| 许可 | ODC-By 1.0（可商用） |
| 下载 | `huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset --include "sample/10BT/*"` |

- 只需 `sample/10BT/` 一个子集（**10 B token / 约 40 GB**），配上 2 个 epoch 就是 20 B。
- **教育质量过滤过**，小模型上表现显著优于原始 CommonCrawl。

### 🥈 方案 2：SkyPile-150B（中文）——中文首选

| 项 | 值 |
|---|---|
| HuggingFace | **`Skywork/SkyPile-150B`** |
| 规模 | **150 B token**（取 18 B 即可，约 12 %） |
| 格式 | parquet |
| 许可 | 需同意条款（**gated**，用 HF 账号申请，通常即时通过） |
| 下载 | `huggingface-cli download Skywork/SkyPile-150B --repo-type dataset` |

- 中文网页语料，去重与质量过滤都做过。
- 注意：**gated**，需要先在网页上点同意。

### 🥉 方案 3：WanJuan-1.0（中英混合）——一条命令拿全

| 项 | 值 |
|---|---|
| HuggingFace | **`opendatalab/WanJuan1.0`** |
| 规模 | **1 TB+ 原始**，中英双语多来源 |
| 格式 | parquet |
| 许可 | Apache-2.0 / CC-BY-4.0（分来源） |

- 上海 AI Lab 出品，含网页/书籍/论文/代码等子集。
- 想同时要中英文就选它。

---

## 三、备选与补充

| 数据集 | HF 路径 | 规模 | 用途 |
|---|---|---|---|
| **CulturaX** | `uonlp/CulturaX` | 6.3 T token / 167 语种 | 多语种；`zh` 子集约 500 B token |
| **Chinese-FineWeb-Edu** | `opencsg/Chinese-FineWeb-Edu` | 约 100 B token | 中文教育质量过滤版，**中文里质量最高的之一** |
| **MNBVC** | `liwu/MNBVC` | 40 TB+ | 中文超大规模，但需大量清洗 |
| **The Pile** | `EleutherAI/pile` | 825 GB / 300 B token | 英文经典混合，多样性好 |
| **Dolma** | `allenai/dolma` | 3 T token | 完全开源可复现，含数据配方 |
| **FineWeb**（非 Edu） | `HuggingFaceFW/fineweb` | 15 T token | 规模最大，质量略低于 Edu |
| **RedPajama-1T** | `togethercomputer/RedPajama-Data-1T` | 1 T token | 经典复现基线 |

**组合建议**：**FineWeb-Edu 12 B + SkyPile-150B 6 B**（英文为主、中文保底），
或纯中文场景用 **SkyPile-150B 18 B**。

---

## 四、不要选的数据集（明确排除）

| 数据集 | 为什么不要 |
|---|---|
| **C4** | 过滤激进、质量已被 FineWeb 全面超越 |
| **OpenWebText** | 仅 40 GB / 9 B token，勉强够但质量不如 FineWeb |
| **Wikipedia dump** | 仅约 4 B token，**单独用严重欠训** |
| **任何 < 5 B token 的集合** | 训 0.9B 会欠训，浪费算力 |
| **合成/模板语料（含我们自己的 L5）** | 多样性上限 = 模板数，**不能用来「对标 transformer」** |

---

## 五、下载后的处理链（仓库已有的工具）

```
1) 下载 parquet 到 D:\TaoVm\data\pretrain\raw\
2) 训练新 BPE 词表（词表 32768/65536）
3) export_repair_bpe_train.exe  parquet -> train.bin（TLP2 格式）
4) 训练：train_noffn_probe_delta.exe（H2R）或 ds3 版
```

**已知的两个改动点（必须先做）**：

1. **词表冻结校验**：`train_noffn_probe.cu` 第 42 行硬编码
   `require(th=="34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333")`
   —— 换词表后**必然失败**，需改成可配置或更新哈希。
2. **文档数上限**：第 49 行 `docs.size()<=65536` —— 18 B token 语料远超此值，
   必须**改成流式/分片训练**，不能一次性读入。

---

## 六、训练器侧的三个硬改动（长训必需）

排查 `src/train_noffn_probe.cu` 后确认：

| # | 问题 | 现状 | 必需改动 |
|---|---|---|---|
| 1 | **步数上限** | 第 37 行 `updates 1..1000` | 长训远超 1000，需放开 |
| 2 | **无续训路径** | 恒 `initialize(Config{},...)` 从零初始化；doc 05 §4.5 已核实 | **实现从检查点恢复**（含 Adam 动量/方差），否则崩溃即前功尽弃 |
| 3 | **检查点间隔** | 第 122 行 `tr.steps%50u==0` | 已满足「每 100 步保存」；若要严格 100 可改 |

> 第 2 条是**长训的前提**。当前训练器**完全不支持续训**，跑 18 B token 需要数天，
> 中途任何中断都会全部丢失。

---

## 七、显存路线（见 doc 17 §2.4–2.5）

**8 GB 显存装不下 0.9B**：现有训练器实占 **20 B/参数**（master + moment + variance + grad + 生效权重），
0.9B 需 **≈ 18 GB**。

**推荐路径**：优化器状态 CPU offload（`master`/`moment`/`variance` 放 47 GB 主机内存）
+ bf16 梯度 → GPU 侧 **≈ 5.4 GB**，可行。

---

## 八、待你确认

1. **语言取向**：纯英文 / 纯中文 / 中英混合？（决定下 FineWeb-Edu 还是 SkyPile）
2. **磁盘余量**：D: 盘能腾出 ≥ 200 GB 吗？
3. **HF 网络**：能直连 huggingface.co 吗？（国内需镜像 `hf-mirror.com`）
4. **是否接受 gated 数据集**：SkyPile-150B 需要 HF 账号点同意。
