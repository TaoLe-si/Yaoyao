# 夭夭 Yaoyao 0.1.1

夭夭是原生 C++ / CUDA 三值双状态语言模型。**0.1.1** 的正式算子为 `dual-state-4-noffn-delta-mem-input-sqrt-d`：GPU 训练、CPU 常驻权重贪心解码；每层一个门控短状态向量加一个**增量规则矩阵记忆**（delta-rule matrix memory），无 FFN、无注意力、无 KV cache。

相对 0.1.0 的变化：记忆分支由"门控向量状态 + read.m 读出"改为"秩一增量写入的矩阵记忆 $M\in\mathbb{R}^{m\times dk}$ "；层数 8 → 2；训练/推理激活统一为同一三阶 Padé 近似（doc 24）。

## 架构

| 项目 | 0.1.1 配置 |
|---|---:|
| 层数 $L$ | 2 |
| 主通道 $d$ | 512 |
| 短状态 $s$ | 128 |
| 记忆矩阵 $M\in\mathbb{R}^{m\times dk}$ | 512 × 64 |
| BPE 词表 $V$ | 16,384 |
| 三值矩阵元素 | 9,502,720 |
| 浮点偏置／归一化参数 | 20,482 |
| 合计参数 | **9,523,202** |
| 每层循环状态 | 32,896 float32（128.5 KiB） |
| 每会话循环状态（2 层） | 65,792 float32（**257 KiB，与上下文长度无关**） |
| 配置字段 $e$ | 1,024（序列化兼容字段，无 FFN 计算含义） |

权重算子身份 `dual-state-4-noffn-delta-mem-input-sqrt-d`；DSB/SCP schema 与 0.1 互不兼容，禁止混用旧解码器或续用旧检查点。训练与推理遵守同一算子（R4），激活统一守卫见 `act_parity_test`。

## 完整数学定义

记 token 序列 $z_1,z_2,\dots$ ， $t$ 为位置， $\ell=1,\dots,L$ 为层（以下省略层标）。矩阵均为"输出维 × 输入维"， $\odot$ 为逐元素积。

### 输入与归一化

共享嵌入 $E\in\mathbb{R}^{V\times d}$ （三值，行尺度 $\alpha$ ）。输入缩放与 RMSNorm：

$$x_t^{(0)}=\sqrt{d}\,E_{z_t,:}^{\mathsf T},\qquad
R_\gamma(x)=\gamma\odot x\,\rho,\qquad
\rho=\Big(\tfrac{1}{n}\lVert x\rVert_2^2+10^{-5}\Big)^{-1/2}.$$

每层入口 $h_t=R_{\gamma_{in}}\big(x_t^{(\ell)}\big)$ 。

### 短状态：门控凸组合更新

$$u_t=W_{sx}h_t+W_{ss}s_{t-1}+b_s,\qquad
a_t=G_{sx}h_t+G_{ss}s_{t-1}+b_{gs},$$

$$s_t=s_{t-1}+\sigma(a_t)\odot\big(\tanh(u_t)-s_{t-1}\big)
=(1-\sigma(a_t))\odot s_{t-1}+\sigma(a_t)\odot\tanh(u_t).$$

部署激活为三阶 Padé 近似并夹紧：

$$\widehat{\tanh}(x)=\frac{x(27+x^2)}{27+9x^2},\quad |x|>3\Rightarrow\pm1;\qquad
\widehat{\sigma}(x)=\tfrac12\big(1+\widehat{\tanh}(x)\big).$$

（记忆门 $\beta$ 与损失中的 sigmoid 保持精确式，见"训练与推理同算子"。）

### 记忆：增量规则矩阵写入

每层维护运行时矩阵 $M\in\mathbb{R}^{m\times dk}$ ，会话开始时 $M_0=0$ 。由 $h_t$ 投影：

$$k_t=W_k h_t\in\mathbb{R}^{dk},\quad q_t=W_q h_t\in\mathbb{R}^{dk},\quad
v_t=W_v h_t\in\mathbb{R}^{m},\quad
\beta_t=\sigma\big(\langle w_\beta,h_t\rangle+b_\beta\big)\in(0,1).$$

键归一化、先读、秩一写入、后读：

$$\hat k_t=\frac{k_t}{\lVert k_t\rVert_2+10^{-6}},\qquad
a_t=M_{t-1}\hat k_t,\qquad
M_t=M_{t-1}+\beta_t\big(v_t-a_t\big)\hat k_t^{\mathsf T},\qquad
o_t=M_t q_t.$$

即逐行 $M_t[i,:]=M_{t-1}[i,:]+\beta_t\big(v_t[i]-a_t[i]\big)\hat k_t$ 。

### 读出与残差

$$r_t=W_{rs}s_t+o_t,\qquad
x_t^{(\ell+1)}=x_t^{(\ell)}+R_{\gamma_{read}}(r_t).$$

无 FFN 分支。 $L$ 层后共享头输出：

$$\ell_t=E\,R_{\gamma_f}\big(x_t^{(L)}\big)+b_v,\qquad
p(z_{t+1}=j\mid z_{\le t})=\frac{e^{\ell_{t,j}}}{\sum_i e^{\ell_{t,i}}}.$$

### 推理端重复惩罚

对**本回合已生成**的每个 token 施加乘法惩罚（提示不参与，保护逐字抄写任务）：

$$\ell_{t,j}\leftarrow\frac{\ell_{t,j}}{\pi}\ \ (\ell_{t,j}\ge0,\ j\in S_t),\qquad
\ell_{t,j}\leftarrow \ell_{t,j}\cdot\pi\ \ (\ell_{t,j}<0,\ j\in S_t),\qquad
S_t=\{z_{\le t}\text{ 中本回合已生成的 token}\}.$$

默认 $\pi=1.3$ ， $\pi=1$ 等价关闭。 $S_t$ **只增不减**：token 出现过一次即进入惩罚集，重放时再次命中，因此**对每个 token 都有效，不是固定窗口**；\`--rep-win W\` 可退回旧滑窗行为（仅惩罚最近 $W$ 个）。

实测（checkpoint step_7573，held-out 200 条， $\pi=1$ 即无惩罚基线）：

| 配置 | 退化复读 | bigramF1 | 乱码 | 空回复 |
|---|---:|---:|---:|---:|
| 无惩罚 | 104/200 | 0.0414 | 9 | 0 |
| 减法 $\lambda=1.3$ / 窗口 32（旧实现） | 22/200 | 0.0515 | 6 | 0 |
| 乘法 $\pi=1.2$ / 全回复 | 7/200 | 0.0556 | 4 | 0 |
| **乘法 $\pi=1.3$ / 全回复（默认）** | **0/200** | 0.0547 | **2** | **0** |
| 乘法 $\pi=1.5$ / 全回复 | 0/200 | 0.0534 | 1 | 0 |

两个条件必须同时成立：(1) 覆盖全部已生成 token；(2) 用**乘法缩放**而非减法。只做 (1) 而用减法会崩坏 —— 常用虚词的 logits 被整体压塌，退化复读反而升至 104/200。长度比 0.81 与零空回复说明改善不是靠缩短输出。

## 数学性质（严格论证）

**命题 1（短状态有界）。** $s_0=0$ 时对一切 $t$ 有 $\lVert s_t\rVert_\infty\le 1$ 。
*证*： $\widehat{\sigma}\in(0,1)$ 、 $|\widehat{\tanh}|\le 1$ （夹紧区取等号），故 $s_t[j]$ 是 $s_{t-1}[j]$ 与 $\widehat{\tanh}(u_t)[j]\in[-1,1]$ 的凸组合。对 $t$ 归纳即得。∎

**命题 2（写入是一步归一化最小二乘梯度下降）。** 记 $f(M)=\tfrac12\lVert M\hat k_t-v_t\rVert_2^2$ ，则 $\nabla_M f=(M\hat k_t-v_t)\hat k_t^{\mathsf T}$ 。增量规则恰为

$$M_t=M_{t-1}-\beta_t\nabla_M f,\qquad \eta=\beta_t.$$

即 $\beta_t(h)$ 是由输入调制的**逐位置学习率**。更新后沿 $\hat k_t$ 的读出满足 $M_t\hat k_t=(1-\beta_t\tau)M_{t-1}\hat k_t+\beta_t\tau\,v_t$ ，其中 $\tau=\hat k_t^{\mathsf T}\hat k_t=\lVert k\rVert^2/(\lVert k\rVert+\varepsilon)^2$ 。 $\varepsilon=10^{-6}\ll\lVert k\rVert$ 时 $\tau\approx1$ ，读出以比例 $\beta_t$ 向 $v_t$ 移动； $\tau<1$ 的精确值不破坏命题 3。

**命题 3（按方向寻址，正交不互扰）。** 若 $\hat k_t\perp\hat k\'$ ，则对任意 $M$ ：

$$M_t\hat k\'=M\hat k'+\beta_t\big(v_t-M\hat k_t\big)\big(\hat k_t^{\mathsf T}\hat k\'\big)=M\hat k\'.$$

写入只作用于与 $\hat k_t$ 平行的读出方向；相同或相近方向按命题 2 覆写更新。这是精确等式（不依赖 $\varepsilon$ ）。记忆因此是**以 key 方向为地址的关联存储**：可寻址、可覆写、容量由 $dk$ 维方向空间与歧义冲突决定，**不是无限记忆**。

**命题 4（常数状态与复杂度）。** 每层运行时状态为 $s\in\mathbb{R}^{128}$ 与 $M\in\mathbb{R}^{512\times64}$ ，共 32,896 float32；2 层合计 65,792 float32 = 257 KiB。每 token 计算量（符号加减计）：

$$\underbrace{2(sd+s^2)}_{\text{短状态}}+\underbrace{d(2dk+m)}_{\text{记忆投影}}+\underbrace{3m\,dk}_{\text{读-写-读}}+\underbrace{ds}_{\text{读出}}=655{,}360\,/\text{层},\qquad \text{加头 }Vd=8{,}388{,}608.$$

解码每 token $O(1)$ 、状态 $O(1)$ 、**与上下文长度无关**；训练为分块 TBPTT，每块 $O(\text{块长})$ 。

**参数计数。** 由 schema：

$$P_{\text{tern}}=Vd+L\big[2(sd+s^2)+2dk\cdot d+md+ds\big]=9{,}502{,}720,$$

$$P_{\text{flt}}=V+d+L\big[2s+d+1+2d\big]=20{,}482,\qquad P=9{,}523{,}202.$$

## 三值投影

对 master 权重每行 $w\in\mathbb{R}^n$ 求逐行最优三值重构：

$$\min_{\alpha\ge0,\ q\in\{-1,0,1\}^n}\lVert w-\alpha q\rVert_2^2.$$

固定非零集合 $S$ 时最优尺度与误差为

$$\alpha^{\ast}_S=\frac{1}{|S|}\sum_{i\in S}|w_i|,\qquad
J^{\ast}(S)=\lVert w\rVert_2^2-\frac{\big(\sum_{i\in S}|w_i|\big)^2}{|S|}.$$

按 $|w|$ 降序记前缀和 $A_k=\sum_{i\le k}a_i$ ，取使 $A_k^2/k$ 最大的最小 $k^{\ast}$ （等值取小 $k$ ）， $\alpha^{\ast}=A_{k^{\ast}}/k^{\ast}$ 。这是实数算术下该行重构的全局最优解。离散化用 identity STE：

$$\frac{\partial\mathcal L}{\partial w^{\text{master}}}\approx\frac{\partial\mathcal L}{\partial w^{\text{eff}}},\qquad w^{\text{eff}}=\mathrm{diag}(\alpha)q.$$

训练保存 float master 与 Adam 矩；2-bit 符号存储只是序列化格式。

## 训练与推理同算子（R4）

训练器（GPU）与解码器（CPU）使用同一 $\widehat{\tanh}$ （doc 24）。逐点守卫 `act_parity_test`：前向一致 max **1.192e-07**（1 ULP，nvcc FMA 收缩）、导数对双精度中心差分 max **1.925e-07**、门控更新一致 max **7.749e-07**；修复前 tanh/sigmoid 偏差分别为 2.35e-02 / 1.59e-01（**197,000× / 1,336,000×** 改善）。判据为语义等价 ≤4 ULP。

## 损失与 TBPTT

监督掩码 $\mu_t$ 、 $N=\sum_t\mu_t$ ：

$$\mathcal L=-\frac1N\sum_t\mu_t\log p(z_{t+1}\mid z_{\le t}),\qquad
\frac{\partial\mathcal L}{\partial \ell_{t,j}}=\frac{\mu_t}{N}\big(p_{t,j}-\mathbf 1[j=z_{t+1}]\big).$$

训练用真实历史 token（teacher forcing），生成用自身输出：**低 NLL 不保证不循环**，行为指标必须另测。

## 引用文献与借鉴

**[1] Vaswani et al., 2017.** *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762.

- **借鉴**：自回归语言建模的任务形式（teacher forcing、逐步交叉熵、贪心/温度解码）；嵌入-输出头共享的现代实践；残差流 + 前置归一化的组织方式（[1] 原文为 post-norm，pre-norm 属后续演化）；以及 $1/\sqrt{d_k}$ 缩放传统——我们的嵌入入口 $\sqrt d$ 缩放与之反向呼应，同为控制进入内积/归一化的方差量级。
- **有意剔除**：softmax 注意力与随上下文增长的 KV cache。序列混合函数由 $\mathrm{softmax}(QK^{\mathsf T}/\sqrt{d_k})V$ 换成门控凸组合递归（短状态）与秩一 delta 写入（记忆）的复合，状态 $O(1)$ 。

**[2] Gu & Dao, 2023.** *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.

- **借鉴**：选择性线性递归作为注意力替代的路线合法性；"选择性=输入依赖"在两处落地——短状态门 $\sigma(a_t)$ 与记忆写入强度 $\beta_t(h)$ （命题 2 意义下即逐位置学习率）；常数状态、线性时间的复杂度目标；以及硬件感知的训练重构（其并行 scan 对应我们的分块 TBPTT + slot 批处理图）。
- **差异**：Mamba 状态为结构化 SSM 的对角 $A$ 、步长 $\Delta$ 、输入投影 $B,C$ ；我们不做 SSM 参数化——主记忆是显式矩阵 $M$ 的秩一 delta 写入（关联记忆谱系），短状态更新是 GRU 式凸组合，二者是不同的状态因子分解。

**[3] DeepSeek-AI, 2026.** *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression.* 技术报告（51 页），模型见 [huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)。

- **论点继承并推到端点**：[3] 论证 KV cache 是长上下文推理的主要成本（全局 KV 压至 890 B/token，为前代 1/4）。我们把同一论点推到极限：**没有注意力即没有 KV cache**——上下文状态恒定 257 KiB、增长 0 B/token。代价同样明确：无精确远程检索，上下文必须压缩进固定状态（容量界见命题 3）。
- **CED ↔ 会话协议**：[3] 的因果编码器-解码器让提示的全局 KV 由编码器一次构建、解码只做增量；对应我们的 USER/ASSISTANT/TURN_END 会话协议——提示一次流过循环状态，解码阶段不再重放提示。
- **CSA2 跨层复用 ↔ 层间复用**：[3] 跨层共享全局 KV 与稀疏选择；对应我们的跨层维度复用与 engram 条件记忆提案（docs/architecture-proposals 03/04/13）。其 Engram 组件（稀疏访问条件记忆）与我们 $m\times dk$ 可寻址矩阵记忆是同一思想——把上下文外置为可寻址存储——的两种实现。
- **FP4 ↔ 三值**：[3] 把主 KV 压到 FP4；我们在权重侧走同一压缩谱系更进一步的 $\{-1,0,+1\}\approx1.58$ bit，并同样保持"存储量化、计算用高精度 master"的分离。
- **工程纪律**：[3] 的跨代 A/B 实证与消融对应我们的 R4 同算子铁律、`decode_diag` A/B 与 `act_parity` 永久守卫——结论必须由同一把尺子的对照测量支撑。

## 实证状态（v0.1.1 训练中，如实）

- **真实语料**：alpaca-zh 48,418 docs / 7.86M tokens（对话，TLP2）；wiki-cn 254,107 docs / 181.1M tokens；code 60,477 docs / 158.3M tokens（后两者待混入）。
- **训练**：9 分片课程流式（先精华后混合），lr 阶梯 1e-3→3e-4→1e-4，400 步/分片；跨轮对比用 held-out 400 docs 固定评测（n=200 噪声过大，不足以判定跨轮趋势）。
- **当前（第三轮进行中，shard 3/8）**：真实能力 loop(pen=0)=57.5%、bigramF1=0.0423（n=400 全量 held-out）；部署解码（乘性全回复 π=1.3）loop=2/400、bigramF1=0.0543。
- **跨轮趋势（n=400 全量 held-out，同一评测集，三个检查点都收在 shard 8，无分片混淆）**：

  | 检查点 | loop(pen=0) | bigramF1 | 精确率 | 均长 | 部署 loop | 部署 F1 |
  |---|---:|---:|---:|---:|---:|---:|
  | 第一轮末 (step 3973) | 273/400 (68.3%) | 0.0366 | 0.152 | 76 字 | 7/400 | 0.0530 |
  | 第二轮末 (step 7573) | 230/400 (57.5%) | **0.0423** | **0.156** | 73 字 | 2/400 | **0.0543** |
  | 第三轮末 (step 11173) | **198/400 (49.5%)** | 0.0372 | 0.131 | 68 字 | 4/400 | 0.0488 |

  **两个轴在第三轮分道扬镳（如实记录）**：
  - **重复度持续改善**：用长度无关指标复核，同一回复内 4-gram 自重复 0.522 → 0.444 → **0.394**，8-gram 自重复 0.432 → 0.342 → **0.302**，最重复 8-gram 占比 1.41 → 0.95 → **0.84**。这些指标与长度无关，且长度只降了 10% 而重复度降了 24%，因此"越学越不重复"是**真实效应而非缩短假象**。
  - **相关性在第三轮退步**：精确率 0.152 → 0.156 → **0.131**（−16%），F1 跌破第二轮。精确率按生成侧 bigram 计，不受长度影响。
  - 同时 shard 8 训练 NLL 继续单调下降（2.768 → 2.383 → **2.198**），"见过 vs 没见过"的 F1 差距从第二轮 +9% 重新拉大到 +30%。
  - **判定**：第三轮首次出现"更流畅但更空洞"的分化，即过拟合开始。跨轮结论必须用 n=400（n=200 曾被噪声误导）。
### 是否在背答案？直接检验（in-sample vs held-out）

从训练语料均匀抽 200 条**模型见过**的样本（参考均 229 字），与 200 条从未见过的 held-out（均 216 字）同尺对比三组指标：

| 检查点 | 数据 | bigramF1 | 8-gram 逐字覆盖 | 前缀匹配 |
|---|---|---:|---:|---:|
| 第一轮末 | 见过 | 0.0431 | 0.0003 | 0.5 字 |
| | 没见过 | 0.0365 | 0.0001 | 0.4 字 |
| 第二轮末 | 见过 | 0.0452 | 0.0009 | 0.5 字 |
| | 没见过 | 0.0414 | 0.0001 | 0.6 字 |
| 第三轮中 | 见过 | 0.0428 | 0.0001 | 0.6 字 |
| | 没见过 | 0.0425 | 0.0003 | 0.6 字 |
| 第三轮末 | 见过 | 0.0476 | 0.0002 | 0.4 字 |
| | 没见过 | 0.0365 | 0.0001 | 0.5 字 |

结论：

1. **不是在背答案**。8-gram 逐字覆盖与前缀匹配在两列都贴近零 —— 即便对**见过**的样本，也几乎不逐字复现参考答案。
2. **泛化差距先收窄后重新拉大**：见过 : 没见过的 F1 差距 第一轮 +18% → 第二轮 +9% → **第三轮末 +30%**。第二轮曾在收敛，第三轮末反转为发散 —— 与"精确率退步、训练 NLL 继续下降"一致，指向过拟合开始。
3. **held-out F1 非单调**：0.0365 → 0.0414 → 0.0372（第三轮中曾达 0.0470）。
4. 诚实的边界：本模型体量下逐字复现能力本就极弱，因此"不背诵"同时也受容量限制；它学到的是格式、语气与浅层主题相关，尚未达到内容级能力。

- **已出现**：编号列表/代码块（```python def 与题意对应）、情境正确的拒绝、助手人格。
- **未出现**：数值计算、知识问答、内容组合（loop(pen=0) 未降即其度量）。
- **已知效应**：分片切换使行为指标摆动 ±10 个百分点；结论只取同分片边界对比。NLL 分片间不可比（3.4–4.7 均为正常值域）。

## 构建

- `scripts/build_train_shards.bat` — 训练器（B，唯一训练入口）：`train_shards SHARD_DIR TOK OUT_DIR STEPS [SLOTS WIDTH] [RESUME] [START_SHARD]`
- `scripts/build_h2r_cpu.bat` — 解码器：`h2r_cpu MODEL [--rep-pen F] [--rep-win N]`
- `scripts/build_act_parity.bat` — 激活一致性守卫（每次架构改动必跑）

## 版本历史

- **0.1.1** — 算子升级 `dual-state-4-noffn-delta-mem-input-sqrt-d`：delta 规则矩阵记忆取代向量记忆分支；层数 8→2；训练/推理激活统一（Padé，parity ≤4 ULP）；解码端重复惩罚改为**乘法全回复覆盖**（默认 π=1.3，对每个已生成 token 都生效）；真实语料训练管线（分片课程 + 断点续训 + 新洗牌种子）。
- **0.1.0** — `dual-state-3-noffn-input-sqrt-d`：门控向量双状态，无 FFN；架构与数学基线。
