# DeepSeek-V4.1-Flash 技术报告精读与迁移判定

> 论文：`D:\\Backup\\Downloads\\DeepSeek_V41_Tech_Report.pdf`（51 页，pdfTeX/LaTeX，标题 *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*）
> 抽取文本：用 `tools/extract_pdf.py` 可重新生成（本轮抽取：51 页 / 162,313 字符）。本文件所有【论文】数字均为报告转述。

---

## 一、一句话主旨

报告不是「更快注意力」的论文，而是**当稀疏注意力把算力降下来之后，如何对付转移到 KV 存储/复用/搬运上的瓶颈**：它在**条目大小 × 序列维 × 层维**三个可乘维度上各下一刀，并把所有推理期近似**在训练期原样模拟**。

---

## 二、问题定义与瓶颈转移链

1. 长程 agent 使负载变为 **input-heavy**（工具调用频繁 → 大量 prefill 与上下文复用）。
2. 稀疏注意力（V4 的 CSA/HCA）已把**算力**降下来 → 瓶颈**转移**到 KV 的持久化、复用与搬运。
3. KV 分三类，约束不同：
   - **全局 KV**（main KV + indexer K）：长驻 **HBM**，随序列线性增长；
   - **持久 KV**：为 prefix reuse 落盘（**SSD/主机内存**），受容量与 I/O 带宽约束；
   - **SWA KV**：只服务本地窗口，容量与序列长度无关，但在 V4 部署中占持久 KV 容量近一半。
4. 三个独立成本维度：**算力、HBM 占用、SSD 与互连带宽**。

> 对夭夭的启示：我们用固定状态把序列维**直接消掉了**（每 token 0 字节 KV），代价是失去「按需检索任意历史位置」的能力。论文路线是「保留全局分支再压缩它」，而不是删掉它。

---

## 三、架构总览【论文 §2.1、§4.2.1】

| 项 | 值 |
|---|---|
| 骨干 | 40 层 = **20 层 causal encoder + 20 层 decoder**；d = 5120 |
| 注意力 | 前 2 层纯 SWA；其余层 = **global（CSA2）+ 每层自带 SWA** |
| MoE | 每块都是 MoE：1 shared + 384 routed，专家中间维 2304，每 token 激活 6 个，SwiGLU(clamp 10) |
| 参数 | **552B backbone + 196B Engram** |
| 激活 | **prefill 8B/token，decode 16B/token**（CED 的直接结果） |
| 上下文 | 1M token；原生多模态（ViT，3×3 pixel-unshuffle，约 1344×1344 等效） |
| 其他 | Single-Pass mHC、Engram、DSpark、Hierarchical Sparse Indexer |
| 关键收益 | 同序列长度下 runtime KV ≈ V4-Flash 的 **1/4**，持久 KV ≈ **1/8**；decode FLOPs 随上下文近乎恒定（4K → 1M 仅 +25 %） |

---

## 四、CED：Causal Encoder-Decoder【论文 §2.2】

对 l > L/2（decoder）：

    C_l = H_{L/2} · W^{KV}_l        （KV 条目）
    Z_l = H_{L/2} · W^{Z}_l         （对应的压缩权重）

- 源头是 YoCo（上半层复用下半层 KV）；CED 的改进是**把 KV 的生成深度做深**（从 L/2 层隐状态用「层专属」投影重新生成，而非复用下半层逐层缓存），同时**本地 SWA 仍逐层计算**，保证局部建模深度不降。
- 代价：decoder 的 SWA KV 需额外处理 n_win × L/2 个 token（128 × 20 = 2560）。
- 解法 **Decoder SWA Bounded Replay**：只 prefill prompt 的最后 n_win 个 token（依据：SWA 有效感受野远小于理论窗口）。
- 复杂度 **O(NL) → O(NL/2 + n_win·L/2) ≈ O(NL/2)**。

**给夭夭的可迁移内核**：这是「**让上层少算，把深度让给下层**」的范式；对应候选实现是「浅层 prefill」（见 06 号文档 C5），但**只改善 TTFT，不改善逐 token 吞吐**。

---

## 五、CSA2：Compressed Sparse Attention 2【论文 §2.3】

### 5.1 三个可乘维度

| 维度 | 论文手段 | 夭夭对应物 |
|---|---|---|
| 条目大小 | GQA 减 KV 头；MLA 跨头共享 latent；**FP4 KV** | 三值权重（2 bit/元素）+ 每行 float32 α |
| 序列维 | 每 m 个 token 压成 1 个条目（压缩率 m） | **不存在**（状态与长度无关） |
| 层维 | **跨层复用 main KV / indexer K / Top-K 索引**；或整层替换 | **完全未利用**（8 层 8 份独立参数） |

CSA2 相对 CSA 的简化：去掉相邻压缩条目之间的重叠源；去掉压缩期的绝对位置嵌入；**indexer K 改为由 main KV 投影**（而非从隐状态另走一条压缩路径）。

### 5.2 三种静态模式

| 模式 | main KV | indexer K | Top-K 索引 | 本层自算 |
|---|---|---|---|---|
| Full | 本层生成 | 由本层 main KV 投影 | 本层 indexer Q 全量打分后选出 | main Q + SWA KV + indexer Q |
| Reindex | 复用最近 Full 层 | 复用 | **重新打分**（KV 共享但选择随层变化） | main Q + SWA KV + indexer Q |
| Reuse | 复用 | 复用 | **复用**（最近 Full/Reindex 产生） | 仅 main Q + SWA KV |

要点：**缓存共享与选择复用被解耦**（不引入额外状态）；**静态指派**（不做运行时动态路由，训练/推理路径一致）；被指派为 Full 的 decoder 层其 KV 来自 H_{L/2}。

### 5.3 V4.1 的实际指派（可作指派模板）

- encoder：18 层 CSA2，m = 2，3 组 × 6 层，每组 **首层 Full + 后 5 层 Reuse**；
- decoder：20 层 CSA2，m = 1，5 组 × 4 层：第 1 组 **Full + 3 Reuse**，其余 4 组 **Reindex + 3 Reuse**；
- indexer：32 query 头 × 128 维；attention top-k = **512**；main Q 64 头 × 512 维；query 压缩维 1280；n_win = 128。

---

## 六、Hierarchical Sparse Indexer【论文 §2.3.2】

- decoder 中**第一个 Full 层**在做 Top-K 的同时构造**块级候选池**（每块取块内最大 index 分数，选最高分的若干块：2048 块 × 8 位置 = **16 384 候选**）；
- 之后 **Reindex 层只在该池内打分**（池共享、各层最终 Top-K 可不同）；Reuse 层不索引；
- 效果：对固定池大小，后续 indexer 每 query 代价从 O(上下文) 变**常数**；代价是第一个 Full 层仍全扫；
- **最关键纪律：候选限制在训练与推理中完全一致地施加**。

**给夭夭的迁移**：输出头占 38.6 % MAC，但实测表明**精确剪枝不可能**（见 02 号文档「死路」）→ 只能照论文做成「近似 + 训练期同域」。

---

## 七、FP4 Main KV【论文 §2.4.4】

- V4 已对 indexer Q/K 做 FP4 QAT；V4.1 **扩展到 main KV cache**。FP4 是为**省存储**，不是加速矩阵乘（注意力前先 dequantize，因此可用更精确格式而不依赖硬件支持）。
- 格式：**MXFP4（OCP）E2M1 + 每 16 通道一个 E4M3 scale**，去掉二级全局 scale。
- **范围论证（值得照抄的写法）**：格式上限 448 × 6 = 2688；训练后 RMSNorm 权重幅度约 1 → RMS 归一化后 512 维 latent 的 L2 ≤ √512 ≈ 22.6；RoPE 保范数 → 每通道绝对值 ≤ ≈22.6；实测最大约 10 → 余量充足。
- 工程细节：**RoPE 之后量化**；**SWA KV 保持 FP8**（对量化敏感）；QAT 在**后训练阶段**引入。

**给夭夭的迁移**：① 量化时机（我们在 step 0 就投影三值，论文是在后训练引入 QAT）；② **按范围论证而非拍脑袋**决定格式与 scale；③ dequant 后再计算。

---

## 八、部署：持久 KV 与 SWA Bounded Replay【论文 §3.2】

**V4 的做法与问题**：global KV 与 SWA KV 分开管理、LRU 淘汰；SWA KV 只缓存两个点用于再生与多轮；SSD 配得很大以维持 72 小时命中。问题：**SWA KV 的访问模式与「长保留」策略根本不匹配**（只在会话内分钟级窗口复用）。V4 提出过 Zero SWA Caching（精确重建，重放 L × n_win），生产代价不可接受。

**V4.1 的三条解法**：

1. SWA KV 不再进持久缓存 → 放**每机 10 % 主机 DRAM 的分布式内存池**（分钟级 TTL）；global KV 仍进持久缓存（≥72 小时）；
2. 必然发生的 miss 用 **Encoder SWA Bounded Replay**：只重放最近 n_win 个 token；
3. 这条回退是**基石**：正因为有界重放便宜，才敢把 SWA KV 从持久缓存彻底删除。

**近似的数学含义**：SWA 逐层累积，精确重建需重放 L × n_win；有界重放把 SWA **截断到重放段**——若重放从 s 开始，位置 i 的 query 只能看到 [max(s, i−W+1), i]。

- **Encoder 侧**：重放段只重建 SWA KV、**复用已缓存 global KV（不重算不覆盖）**；重放出的前缀状态是近似的 → 后缀 KV 会**依赖命中位置**（论文明确承认）；
- **Decoder 侧**：CED 下 global KV 来自 encoder 末层，唯一阻碍是 decoder 自己的 SWA KV；只重放最后 n_win 个 token，产出的 decoder SWA KV **只供 decode、不进 prefix cache**；
- 两者「对响应质量影响可忽略」，且**后训练阶段模拟同样的 replay 做 train-aware 适配**。

**给夭夭的迁移**：① 「近似重建 + 训练期模拟」的方法论（对应 06 号文档 C6 bounded-history replay）；② **分层存储与 TTL** 的思路（会话状态热池/冷池）。

---

## 九、效率型架构扩展【论文 §2.4】

### 9.1 Single-Pass mHC（对夭夭有直接 kernel 类比）

- mHC 维护 n 条残差流：`X_{l+1} = B_l X_l + C_l F_l(A_l X_l)`，其中 `(A_l,B_l,C_l) = H(X_l)`；
- 理想「残差更新 + 输入混合 + 系数预测」一次 map 的访存下界：**(2n+2)d**（(n+1)d 读 + (n+1)d 写）；
- V4 的 3 kernel 串行实现是 **(4n+4)d**：因为 **A_l 必须等整个 hidden 维归约完成**，输入混合无法并入第一遍；
- **Single-Pass mHC 把输入混合的系数错开一个 block**（第 l 块用第 l−1 块产生的 A）→ 依赖消失 → 每个 tile 一次读完即可同时做混合与系数累积 → **(2n+2)d**；训练保留旧实现（错位只改变施加的系数），部署融合成 **Mega-mHC**（顺带做 input pre-norm 与 FP8 转换）；性能损失可忽略。

**方法论（对夭夭最有价值的一条）**：用一次**有意的依赖错位**，把不可融合的串行依赖变成可融合。夭夭同 token 内 **s 更新 → m 更新（读 s_t）** 正是同一结构。

### 9.2 Engram（条件记忆）

- 目的：**把记忆从计算里解耦**；实现：tokenizer 压缩 + 多头哈希 + 上下文感知门控 + 多分支融合；
- 配置：**196B 参数**，两个模块放在第 1 与第 14 层（0-indexed，平衡流水阶段显存）；每模块 N-gram 阶数 {2,3,4}、8 个哈希头、每阶总嵌入维 2048；每头一张约 16M 条目、表长为互异素数的表；表与 key/value 投影用 **FP8**；**去掉短因果卷积**（收益不值复杂度）；
- 推理：**确定性寻址** → 后台 RDMA 预取，第一个模块的预取与第一个 Transformer 块重叠；
- 训练：表按行切到 engram 并行组，优化器状态再分片；梯度在 backbone 反向后回传；RL rollout 期间表常驻显存；
- 优化：动量 + Sinkhorn 平衡（见第十节）。

**这是本报告对夭夭最重要的一个部件**——详见 04 号文档。

### 9.3 DSpark（投机解码）

3 层 Transformer drafter（SWA 窗口 128），一次前向并行给出 5 个 draft 位置的 logits，配 Markov head 建模 draft 间依赖、confidence head 预测逐位置接受率；调度器结合引擎吞吐曲线动态选择验证长度。训练与 backbone 解耦（先单独训 drafter，后训练一起训但**梯度不回传 backbone**）。

> **对夭夭不可迁移**：顺序递归状态下验证 k 个 draft token 需要 k 次串行状态更新，没有收益。可借用的只是「昂贵对象两段化」的思想——那个对象是**词表头**而不是 token。

---

## 十、优化器分工【论文 §2.5】

| 参数类别 | 优化器 | 超参 |
|---|---|---|
| 线性变换矩阵（backbone、Engram 投影、视觉-语言投影） | **Muon**（Q/K 用 **head-wise Muon**） | momentum 0.95；wd 0.1；更新矩阵 RMS 重标定到 0.18 |
| RMSNorm 权重、bias、scale 等非矩阵参数 | **AdamW** | β1 0.9、β2 0.95、ε 1e-20、wd 0.1（**norm 有权重衰减，bias/scale 没有**） |
| 嵌入表、token embedding、预测头 | **Nesterov 动量 + Sinkhorn 平衡** | K = 11、τ = 1e-3、ε = 1e-20、γ = 0.18 |

**Sinkhorn 平衡更新（Algorithm 1）**：Nesterov 动量 → 计算行范数 ρ_i 与均值 ρ̄，**把 ρ_i ≤ τ ρ̄ 的行置零**（数值稳定）→ 交替 K 次行/列 L2 归一化（奇数步行、偶数步列）→ 乘 √n 把「单位行 ℓ2 范数」变成「单位行 RMS」→ 用 γ 修正有效学习率以匹配 Adam 的更新幅度。

动机很直白：**给新增的大表挂 Adam 会让优化器状态内存爆掉**（Adam 需 m 与 v 两份缓存），而动量 + Sinkhorn 只需一份动量，且经验上优于 Adam。它近似**等化更新矩阵的行 RMS 与列 RMS**（一行 = 一个 token/identity，一列 = 一个 hidden 特征）。

---

## 十一、训练与后训练【论文 §4、§5】

**预训练**：45T token 多模态语料；batch 固定 100.6M token；warmup 2000 步 → 2.6e-4 保持到 28T → 28T–40T 余弦降到 2.6e-5 → 40T–45T 保持；**从 0 就上稀疏注意力、序列长 64K（无 dense warmup）**；34T 时扩到 1M；文本:多模态 = 7:1；best-fit packing padding ≤ 1e-4。

**后训练（SFT → RL → OPD）**——报告明确说**本版本不引入新的后训练算法**，收益几乎全部来自数据与环境管线：

- 任务三元组 **(problem, environment, verification system)**，按**难度**与**正确性**两维打分，并用两个信号训练模型造更好的任务；
- 多智能体环境构建流水线（判断可容器化/可验证 → 选起点 → 造 fail-to-pass 与 pass-to-pass 评测点 → 自测 → 多 agent 解题 → 独立质检 agent 查错配与可 hack 性 → 修复再验）；
- **DSec**：百万级并发沙箱；分片 + 松弛一致性调度；sub-NUMA 绑定使单机容器密度 1000 → **>2500**；LS 执行类（SCHED_IDLE）抑制干扰；AppArmor + eBPF 防越权（观察到 agent 真的利用 XFS/AppArmor 漏洞、删二进制）；崩溃记失败轨迹并回传 repercussion；
- **异步 RL**：**sample 级派发**（凑够一个 GRPO 组就派下一个 prompt；batch 级会震荡、prompt 级会被长尾卡死）；长度偏置用「按数据集限并发 + 丢弃过早返回的短样本」；off-policy 用「限制最大离线比例 + 陈旧 token 做 loss mask」；训练抢占 rollout 时用 **token 级中断**，并把 KV cache 与专家路由**按 token 粒度持久化**，恢复后续跑；跨 checkpoint 用 **concatenated routing-replay**；
- **模型合并重启 RL**：合并不同 scaffold/配置的 checkpoint 作为下一轮初始化；
- **可控推理投入**：prompt 前置 effort 标量（1–100）；同 (x,b) 组内奖励中心化；长度惩罚 `r_len = -min(C_max, k(b)·ℓ/L_norm)`，其中 **k(b) = k0·exp(−(b−b_min)/τ)**，τ = λΔb；附录 C 在「边际收益 p'(ℓ) ≈ a_x·exp(−ℓ/s_x)」的局部模型下推出**最优长度与 effort 近似仿射**；线上 low/high/max = 50/75/100；
- **评测**：effort 25 → 100，8 项推理基准均值 67.1 → 76.3，DeepSWE v1.1 66.0 → 74.2，Terminal-Bench 2.1 82.4 → 90.6，代价 ≈ **2.5× 输出 token**，且**收益前置**（60–80 档已拿到大部分）；六种 scaffold 迁移良好；多智能体按墙钟 deadline 全面更优（ProgramBench 8h 30.04 % vs 20.39 %），训练奖励含**协作奖励 + 派生延迟惩罚**（执行事件与协作依赖建 DAG，按 prefill/decode 速率与工具耗时加权，取关键路径）。

**论文自承局限**：CSA2 选择错误、bounded replay 的近似状态在**未被测试覆盖的边界**可能退化（重点：长上下文稀疏检索、cache-resumption 边界）；基准接近饱和；评测设施正被能力更强的模型「游戏」。

---

## 十二、论文关键数字速查

| 主题 | 数字 |
|---|---|
| 规模 | 552B backbone + 196B Engram；40 层（20+20），d = 5120；1M 上下文 |
| 激活 | prefill 8B/token，decode 16B/token |
| KV 压缩 | global KV **890 B/token**（V4-Flash 的 1/4）；持久 KV **1/8**；decode FLOPs 4K→1M 仅 +25 % |
| CSA2 配置 | encoder m=2（3 组 ×（1 Full + 5 Reuse））；decoder m=1（5 组：1×（Full+3 Reuse）+ 4×（Reindex+3 Reuse）） |
| 索引器 | indexer 32 头 × 128 维；top-k 512；main Q 64 头 × 512 维；query 压缩维 1280；候选池 2048 块 × 8 = 16 384；n_win = 128 |
| FP4 | MXFP4 E2M1 + 每 16 通道 E4M3 scale（无全局 scale）；范围论证上限 448×6 对实测绝对值约 10；SWA 保持 FP8；RoPE 后量化 |
| Engram | 196B 参数；层 1 与 14；阶数 {2,3,4}；8 头；每阶维 2048；每头约 16M 条目（素数表长）；FP8；无短卷积 |
| mHC | 扩张因子 4；Sinkhorn-Knopp 20 次；访存 (4n+4)d → (2n+2)d |
| 优化器 | Muon(momentum .95, wd .1, RMS→0.18)；AdamW(β .9/.95, ε 1e-20, wd .1)；Sinkhorn(K=11, τ=1e-3, γ=0.18)；Engram lr ×5 |
| 预训练 | 45T token；batch 100.6M；lr 2.6e-4；warmup 2000；28T→40T 余弦到 2.6e-5；64K 起训、34T 扩 1M；文本:多模态 7:1 |
| 后训练 | SFT→RL→OPD（40+ 教师）；effort 1–100；τ 控长度敏感度；25→100 精度 67.1→76.3，token ×2.5 |

---

## 十三、机制 → 本项目迁移判定表

| # | 论文机制 | 夭夭对应物 | 判定 | 迁移方式 |
|---|---|---|---|---|
| 1 | **CSA2 层维复用**（Full/Reindex/Reuse，静态指派） | 8 层完全独立的 s/m 分支与投影 | **可直接迁移** | 层分组：每组 1 Full + 3 Reuse；Reuse 层复用组内权重（R1）或共享输入投影（R2）→ 见 03 号文档 |
| 2 | **Hierarchical Sparse Indexer**（候选池 + 训练期一致） | 输出头（38.6 % MAC、37.3 % 字节） | **类比对，必须近似 + 训练对齐** | 两段式候选池词表头；训练期使用同一受限域 |
| 3 | **FP4 KV + QAT + 范围论证** | 三值 STE + 每行 α | **方法论可迁移** | 量化时机、按范围论证定格式、dequant 后再算、会话状态降精度 |
| 4 | **SWA Bounded Replay + 训练期模拟** | 会话状态 / 上下文截断 | **方法论可迁移** | bounded-history replay（06 号文档 C6） |
| 5 | **CED** | prompt 必须逐 token 走满 8 层 | **类比对（只影响 TTFT）** | 浅层 prefill（06 号文档 C5） |
| 6 | **Single-Pass mHC**（系数错位一个 block） | 同 token 内 s → m 串行依赖 | **可直接迁移（kernel 级）** | m 分支改读 s_{t−1}，使 s/m 融合成一遍 |
| 7 | **Engram** | 20 KiB 固定状态（记忆瓶颈） | **可直接迁移（架构级）** | 哈希 N-gram 条件记忆（04 号文档） |
| 8 | **Sinkhorn 平衡动量** | 共享 embedding+head = 38.5 % 参数 | **可直接迁移** | 替换该张量的 AdamW（06 号文档 C1） |
| 9 | **Muon / head-wise Muon** | 三值 master 矩阵（无注意力头） | **部分可迁移** | 对 master 矩阵试正交化/行归一化更新（消融项） |
| 10 | **reasoning effort（标量条件 + 指数长度惩罚）** | 固定每 token 算力、无思维链 | **类比对** | 深度旋钮：训练期随机丢弃 Reuse 层，部署 8/6/4 档（06 号文档 C3） |
| 11 | **DSpark 投机解码** | 严格顺序递归 | **不可迁移（token 级）** | 思想迁移到词表头（两段 = draft/verify） |
| 12 | **数据/环境管线（任务三元组 + 难度标定）** | probe 语料 + 符号封存实验 | **可直接迁移** | 验证器任务课程（06 号文档 C4） |
| 13 | **BPB 评测** | 仅有监督目标 NLL | **可直接迁移** | 固定验证集上加 bits-per-byte（05 号文档） |
| 14 | **持久 KV 分层（HBM/DRAM 池/SSD + TTL）** | SCP 会话状态 | **可直接迁移** | 会话状态分级存储与精度降级（06 号文档 C2） |
| 15 | MoE、多模态 ViT、DSec、异步 RL 百万沙箱 | — | **不迁移** | 规模与形态不匹配 |

---

## 十四、明确不迁移清单（附理由）

| 论文部件 | 不迁移理由 |
|---|---|
| MoE（1 shared + 384 routed） | 夭夭 body 每 token 仅 13.4M MAC，「无 FFN」是身份（不变量 I1）；条件计算省下的绝对量太小，路由与专家并行都是纯负担。 |
| 多模态 ViT / pixel-unshuffle | 与三值双状态语言模型定位无关，数据与算力预算都不支持。 |
| DSpark token 级投机解码 | 顺序递归下验证 k 个 draft 仍需 k 次串行状态更新，**零收益**。 |
| FP4 main KV / FP8 SWA KV / 890 B/token | 夭夭没有 KV cache，数字不可比；只借用方法论。 |
| head-wise Muon（针对 attention Q/K） | 没有注意力头；只有「按行/按组预处理」的一般思想可试。 |
| 1M 上下文、EPD 分离、DSec、异步 RL、模型合并重启、Agent Team | 规模与资源不匹配；其中「token 级中断 + 状态持久化 + 按样本回收」的思想可映射到 STOP 文件与 SCP 检查点（项目已有雏形）。 |
| 直接套用论文的收益倍数 | 论文的 prefill −50 %、KV ÷4、持久 ÷8 都来自它自己的成本结构；夭夭必须用 05 号文档的账本口径重新推导。 |
