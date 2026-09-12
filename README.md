# 夭夭 Yaoyao 0.1.1

夭夭是原生 C++ / CUDA **三值双状态语言模型**。它**不是 Transformer**：没有 softmax 注意力，没有序列 $`QK^{\mathrm{T}}V`$，没有随上下文增长的 KV cache。

正式算子名：`dual-state-4-noffn-delta-mem-input-sqrt-d`。

- **训练**：GPU（`train_shards`），分片流式、助手位监督、三值投影 + AdamW。
- **解码**：CPU 常驻权重贪心（`h2r_cpu`），每步只更新固定大小的循环状态。
- **每层状态**：门控短向量 $`s\in\mathbb{R}^{128}`$ + **H2R 增量规则矩阵记忆** $`M\in\mathbb{R}^{512\times 64}`$。

Schema 里的 `mem.key` / `mem.query` / `mem.value` 是关联记忆的**写方向、读方向、载荷**，不是注意力头。

发布权重与词表在 [`release/`](release/)（约 2.5 MB 的 DSB2 + `tok_qa.bbp`）。与相关工作的对照见第 7 节，试跑见第 9 节，训练配比见第 10 节。

---

## 1. 配置与参数

| 项目 | 值 |
|---|---:|
| 层数 $`L`$ | 2 |
| 残差通道 $`d`$ | 512 |
| 短状态 $`s`$ | 128 |
| 记忆 $`M\in\mathbb{R}^{m\times d_k}`$ | $`512\times 64`$ |
| BPE 词表 $`V`$ | 16,384（字节基 + 16,123 次合并，特殊号 256–260） |
| 三值矩阵元素 | 9,502,720 |
| 浮点偏置 / 归一化 | 20,482 |
| **合计参数** | **9,523,202** |
| 每层循环状态 | 32,896 个 float32（128.5 KiB） |
| 每会话（2 层） | 65,792 个 float32（**257 KiB，与上下文长度无关**） |
| 配置字段 $`e`$ | 1,024（序列化兼容；本算子无 FFN） |

权重身份与 0.1.0 的向量记忆 schema **不兼容**，禁止混用旧解码器或旧检查点。训练与推理必须同一算子（R4）。

---

## 2. 完整前向定义

记 token 序列 $`z_1,z_2,\dots`$，时刻 $`t`$，层 $`\ell=1,\dots,L`$（下文省略层标）。矩阵一律「输出维 $`\times`$ 输入维」。$`\odot`$ 为逐元乘。

### 2.1 嵌入与 RMSNorm

共享嵌入 $`E\in\mathbb{R}^{V\times d}`$（三值，第 $`j`$ 行尺度 $`\alpha_j`$）。入口缩放与 RMSNorm：

```math
x_t^{(0)}=\sqrt{d} E_{z_t,:}^{\mathrm{T}},\qquad
R_\gamma(u)=\gamma\odot u \rho,\qquad
\rho=\Bigl(\tfrac{1}{n}\lVert u\rVert_2^2+10^{-5}\Bigr)^{-1/2}.
```

每层入口 $`h_t=R_{\gamma_{\mathrm{in}}}(x_t^{(\ell)})`$。$`\sqrt{d}`$ 把三值行从 $`O(1)`$ 抬到与通道维匹配的 RMS 量级，对应注意力里 $`1/\sqrt{d_k}`$ 的**反向**方差控制，而不是注意力本身。

### 2.2 短状态：门控凸组合

```math
u_t=W_{sx}h_t+W_{ss}s_{t-1}+b_s,\qquad
a_t=G_{sx}h_t+G_{ss}s_{t-1}+b_{gs},
```

```math
s_t=s_{t-1}+\sigma(a_t)\odot\bigl(\tanh(u_t)-s_{t-1}\bigr)
=(1-\sigma(a_t))\odot s_{t-1}+\sigma(a_t)\odot\tanh(u_t).
```

部署默认用三阶 Padé 并夹紧（训练器同一式，见第 6 节）：

```math
\widehat{\tanh}(x)=\frac{x(27+x^2)}{27+9x^2},\quad |x|>3\Rightarrow\pm 1;\qquad
\widehat{\sigma}(x)=\tfrac12\bigl(1+\widehat{\tanh}(x)\bigr).
```

记忆写入强度 $`\beta_t`$ 仍用**精确** sigmoid（数值域 $`(0,1)`$，不走 Padé）。

### 2.3 记忆：增量规则（先写后读）

每层一张运行时矩阵 $`M\in\mathbb{R}^{m\times d_k}`$，会话开始 $`M_0=0`$。由同一 $`h_t`$ 投影：

```math
k_t=W_k h_t\in\mathbb{R}^{d_k},\quad
q_t=W_q h_t\in\mathbb{R}^{d_k},\quad
v_t=W_v h_t\in\mathbb{R}^{m},\quad
\beta_t=\sigma\bigl(\langle w_\beta,h_t\rangle+b_\beta\bigr)\in(0,1).
```

键归一化、秩一写入、再读出：

```math
\hat k_t=\frac{k_t}{\lVert k_t\rVert_2+10^{-6}},\qquad
a_t=M_{t-1}\hat k_t,
```

```math
M_t=M_{t-1}+\beta_t(v_t-a_t)\hat k_t^{\mathrm{T}},\qquad
o_t=M_t q_t.
```

逐行即 $`M_t[i,:]=M_{t-1}[i,:]+\beta_t(v_t[i]-a_t[i])\,\hat k_t`$。这是 **H2R / delta-rule 关联记忆**，不是注意力：$`\hat k`$ 是写地址，$`q`$ 是读地址，$`v`$ 是载荷，$`\beta`$ 是这一步的写入学习率。

解码实现用代数恒等式把「先算 $`M\hat k`$、再写、再算 $`Mq`$」收成对每行一次扫描：

```math
o_t=M_{t-1}q_t+\beta_t(v_t-M_{t-1}\hat k_t) (\hat k_t^{\mathrm{T}}q_t).
```

$`M`$ 的流量从 3 次降为 2 次；行互不相交，任意线程划分与串行 **bitwise 一致**。

### 2.4 读出、残差、词表头

```math
r_t=W_{rs}s_t+o_t,\qquad
x_t^{(\ell+1)}=x_t^{(\ell)}+R_{\gamma_{\mathrm{read}}}(r_t).
```

无 FFN。$`L`$ 层后绑定嵌入作分类头：

```math
\ell_t=E R_{\gamma_f}\bigl(x_t^{(L)}\bigr)+b_v,\qquad
p(z_{t+1}=j\mid z_{\le t})=\frac{e^{\ell_{t,j}}}{\sum_i e^{\ell_{t,i}}}.
```

贪心解码 **不物化** 整段 logits，只在行上做 argmax。特殊号 256/257/258（BOS/USER/ASSISTANT）永不作为生成 token。

### 2.5 会话协议（TLP2）

| id | 符号 | 监督 |
|---|---|---|
| 256 | BOS | 否 |
| 257 | USER | 否 |
| 258 | ASSISTANT | 否 |
| 259 | TURN_END | 若落在助手段则为是 |
| 260 | EOS | 若落在助手段则为是 |

用户字节的 `loss=false`，助手字节与助手侧 `TURN_END`/`EOS` 的 `loss=true`。解码：`BOS`（仅新会话）→ `USER` → 提示 → `TURN_END` → 从 `ASSISTANT` 起贪心，直到 259/260 或 `max_out`。

---

## 3. 定理

以下均对单层书写；多层只是把残差串起来，证明不变。

### 定理 1（短状态 $`\ell_\infty`$ 有界）

设 $`s_0=0`$，激活为 $`\widehat{\sigma}\in(0,1)`$、$`|\widehat{\tanh}|\le 1`$（夹紧区取等）。则对一切 $`t`$ 与坐标 $`j`$，

```math
|s_t[j]|\le 1.
```

**证明.** 对 $`t`$ 归纳。$`t=0`$ 显然。坐标互不耦合，只看标量 $`s\leftarrow(1-\sigma)s+\sigma u`$，其中 $`\sigma\in(0,1)`$、$`|u|\le 1`$。这是 $`s`$ 与 $`u`$ 的凸组合，故 $`|s'|\le\max(|s|,|u|)\le 1`$。∎

因此短状态不能靠发散幅值「记住」任意历史，容量在方向与门控时间尺度上，不在范数上。

### 定理 2（写入是一步加权最小二乘梯度）

记 $`f(M)=\frac12\lVert M\hat k_t-v_t\rVert_2^2`$。则

```math
\nabla_M f=(M\hat k_t-v_t)\hat k_t^{\mathrm{T}},\qquad
M_t=M_{t-1}-\beta_t\nabla_M f\big|_{M_{t-1}}.
```

**证明.** 对矩阵变量，$`\mathrm{d}f=\langle M\hat k-v,\,(\mathrm{d}M)\hat k\rangle=\langle(M\hat k-v)\hat k^{\mathrm{T}},\,\mathrm{d}M\rangle`$，即得梯度。代入增量规则右边 $`\beta_t(v-M\hat k)\hat k^{\mathrm{T}}=-\beta_t\nabla_M f`$。∎

**推论.** $`\beta_t(h_t)`$ 是输入调制的**逐步学习率**。沿写地址的读出：

```math
M_t\hat k_t=(1-\beta_t\tau)M_{t-1}\hat k_t+\beta_t\tau v_t,\qquad
\tau=\hat k_t^{\mathrm{T}}\hat k_t=\frac{\lVert k_t\rVert_2^2}{(\lVert k_t\rVert_2+\varepsilon)^2}.
```

$`\varepsilon=10^{-6}\ll\lVert k\rVert`$ 时 $`\tau\approx 1`$，该方向以比例 $`\beta_t`$ 被拉向 $`v_t`$。

### 定理 3（正交地址不互扰）

若 $`\hat k_t\perp \hat k'`$，则对任意 $`M_{t-1}`$：

```math
M_t\hat k'=M_{t-1}\hat k'.
```

**证明.** $`M_t\hat k'=M_{t-1}\hat k'+\beta_t(v_t-M_{t-1}\hat k_t)(\hat k_t^{\mathrm{T}}\hat k')`$，正交使第二项为零。等式不依赖 $`\varepsilon`$。∎

因此 $`M`$ 是 **以 key 方向为地址的有限关联存储**：可写、可覆写、容量由 $`d_k`$ 维球面的可分辨方向与碰撞决定，**不是无限记忆**，也不是「压缩版注意力」。

### 定理 4（状态与每步计算均为 $`O(1)`$）

每层状态 $`|s|+|M|=128+512\cdot 64=32{,}896`$ 个 float。两层合计 257 KiB。每 token 乘加（一层）：

```math
2(sd+s^2)+d(2d_k+m)+2m d_k+ds=655{,}360.
```

拆开：短状态候选/门 $`2(sd+s^2)`$，$`k,q,v`$ 投影 $`d(2d_k+m)`$，融合写读 $`2m\,d_k`$，`read.s` 为 $`ds`$。

词表头 $`Vd=8{,}388{,}608`$，约占逐步计算的 **86%**。上下文变长时这两项都不涨。训练是分块 TBPTT：块内 $`O(W)`$，状态在块边界作为槽位状态延续，不存 KV。

### 定理 5（逐行最优三值重构）

对 master 行 $`w\in\mathbb{R}^n`$，

```math
\min_{\alpha\ge 0, q\in\lbrace -1,0,1\rbrace^n}\lVert w-\alpha q\rVert_2^2.
```

固定非零支撑 $`S`$ 时，$`\alpha_S^\ast=\frac1{|S|}\sum_{i\in S}|w_i|`$，最优值 $`J^\ast(S)=\lVert w\rVert_2^2-(\sum_{i\in S}|w_i|)^2/|S|`$。故全局最优等价于在 $`|w|`$ 降序前缀上最大化 $`A_k^2/k`$。实现按幅值降序、下标升序的 bitonic 排序取最小最优 $`k^\ast`$（等值取更短支撑）。这是实数算术下该行的全局最优，不是启发式舍入。

反传对离散 $`q`$ 用 identity STE：$`\partial\mathcal L/\partial w^{\mathrm{master}}\approx\partial\mathcal L/\partial w^{\mathrm{eff}}`$，$`w^{\mathrm{eff}}=\mathrm{diag}(\alpha)q`$。磁盘上的 2-bit 符号只是序列化；Adam 走 float master。

### 定理 6（参数计数）

由 schema 直接加和：

```math
P_{\mathrm{tern}}=Vd+L\bigl[2(sd+s^2)+2d_k d+md+ds\bigr]=9{,}502{,}720,
```

```math
P_{\mathrm{flt}}=V+d+L(2s+d+1+2d)=20{,}482,\qquad P=9{,}523{,}202.
```

---

## 4. 它是如何学习的

### 4.1 目标

监督掩码 $`\mu_t\in\{0,1\}`$（仅助手侧 token），$`N=\sum_t\mu_t`$：

```math
\mathcal L=-\frac1N\sum_t\mu_t\log p(z_{t+1}\mid z_{\le t}).
```

对 logits 的梯度即 masked softmax 交叉熵：

```math
\frac{\partial\mathcal L}{\partial \ell_{t,j}}=\frac{\mu_t}{N}\bigl(p_{t,j}-\mathbf{1}[j=z_{t+1}]\bigr).
```

实现见 `ds_ce_batch`：无监督槽位梯度为 0，有监督槽位做稳定 log-sum-exp。

### 4.2 Teacher forcing 与 TBPTT

训练喂**真实**历史 token，不是自己的采样。循环状态 $`(s,M)`$ 在时间上展开；实现上按宽 $`W=32`$ 的图块反向，槽位（默认 32）之间并行。文档边界 `reset` 把该槽的 $`s,M`$ 清零。这是截断 BPTT，不是 Transformer 的全序列注意力反向。

**低训练 NLL 不蕴涵不循环。** 生成用自身 token，训练用金标 token，分布偏移是一阶事实。行为必须另测（空回复、复读、bigram F1、CoT 精确命中）。

### 4.3 优化器

AdamW，$`\beta_1=0.9`$、$`\beta_2=0.999`$，主权重衰减 $`0.01`$（仅三值 master），$`\varepsilon=10^{-8}`$。梯度先按监督计数与全局 L2 范数归一，再可叠加 `TAO_GRAD_CLIP`（默认 1.0：若 $`\lVert g\rVert>c`$ 则等效缩小步长）。学习率：20 步线性 warmup，然后从 `TAO_LR_DECAY_START` 起余弦到 `TAO_LR_MIN`。

三值矩阵每步：`master --AdamW--> master'`，再 `project_sorted` 得到生效权重 $`w^{\mathrm{eff}}`$ 供下一前向。STE 把对 $`w^{\mathrm{eff}}`$ 的梯度记到 master。

### 4.4 分片课程

`train_shards` 一次只驻留一个 TLP2 分片。权重与 Adam 矩跨分片连续。`opt_state.bin`（magic `TAOOPT02`）保存 master / moment / variance / steps，可增训到**新目录**（禁止覆盖已有 `OUT_DIR`）。

词表默认冻结为历史 digest；本仓库问答词表 `build/tok_qa.bbp`（digest `13fd68ba…dd75`，16,123 merges）需 `TAO_ALLOW_TOKENIZER=1`。

### 4.5 字节 BPE

词表 = 256 字节 + 5 个特殊号 + 16,123 次 pairwise merge。`encode` 与 merge 表是纯函数；语料导出用 presence-skip + ping-pong 缓冲，避免每个 merge 趟分配。

---

## 5. 解码与优化（非注意力路线）

优化只允许三条轴：**少算、结果复用、算存分离**。禁止把 Transformer 的 KQV / KV cache / 层次稀疏注意力接到本算子上。

### 5.1 已经写进代数的复用

- $`s_t,M_t`$ 本身就是跨 token 的复用状态。
- 同一 $`h_t`$ 生成 $`k,q,v,\beta`$，没有第二套「注意力投影」。
- $`o=Mq`$ 用定理 2 的融合式，避免第三次扫 $`M`$。
- 绑定嵌入：分类头就是 $`E`$，不另存一份 $`V\times d`$。

### 5.2 CPU 热路径（Zen 4 AVX-512）

词表头约占 86% MAC。生产解码（`/arch:AVX512`）：

1. **分档 VNNI**：仅 $`\mathrm{rows}\cdot\mathrm{cols}\ge 10^6`$ 的矩阵（词表头）走 `vpdpbusd`；层内小矩阵走 4 行打包 float 点积。`TAO_VNNI=0` 回到 float 头，逐步 greedy 与旧 checksum 一致。
2. **`vnni8`**：8 个词表行共享一次 64B 的量化激活加载。
3. **线程池阈值 $`2^{20}`$**：只有词表头唤醒 8 线程；`group<4>` 的 $`s`$ 与 `mem.value`（恰好 262,144 MAC）走单线程，避免同步比计算还贵。行不切开，阈值只改调度，**bitwise 一致**。
4. **`RecurBuf`**：$`x,h,k,q,v,o,r`$ 常驻复用，避免每 token 堆分配把 257 KiB 的 $`M`$ 挤出缓存。
5. **激活量化** `packus` 写回；$`M`$ 行预取。
6. **固定窗口进 L2**：每层 $`s|M`$ 收成一块 64 字节对齐的连续区（`LayerState`，约 128.5 KiB）。每步用 `T1` 把整段会话状态预取进 L2；词表头权重走 `NTA`，减少 8.4 MiB 头把 $`M`$ 从 L2 挤出。这不是 Transformer 的滑动窗口注意力：容量恒定，只是把已有的 $`O(1)`$ 状态钉在缓存里。

实测（Ryzen 9 7940H，8 线程，512 forced 步，`build/L1_qa_cot/final.dsb`，默认 VNNI）：均值约 **5.08×10³ tok/s**，好轮约 **5.37×10³**（约 0.19 ms/token）。`TAO_VNNI=0` 时 `step_1600` 的 `last_tok=823`，与改布局前一致。吞吐与上下文长度无关。

钉住 257 KiB **几乎不改墙钟**：窗口本来就进得了 1 MiB L2 / 16 MiB L3，逐步时间的 86% 仍是精确 greedy 头扫词表。TFLA 式分块若要再挖，对象是 **mLSTM 的 $`C`$ ≡ 我们的 $`M`$**（约每层 128 KiB），不是注意力 KV。

### 5.3 重复惩罚（生成，不是训练）

默认 $`\lambda=1.0`$，`max_out=64`。惩罚作用于**本回复已生成的每一个 token**（无窗口）：

- **频率**：每出现一次，该 id 的 logit 再减 $`\lambda`$。旧实现只在第一次出现时减一次，复读「中国中国…」几乎不受影响。
- **3-gram 硬阻断**：若继续某 token 会闭合已出现的 3-gram，将其 logit 减去 $`10^6`$。
- **连写 4 次同一 token** 则收束为 `TURN_END`。

$`\lambda=0`$ 关闭加减与阻断，与无惩罚逐步一致。命令行：`h2r_cpu MODEL [--rep-pen F] [--max-out N]`；stdin：`/rep-pen`、`/max-out`、`/config`。`--rep-win` 已删除。

---

## 6. 训练与推理同算子（R4）

短状态的 $`\widehat{\tanh}/\widehat{\sigma}`$ 在 GPU 训练核与 CPU 解码里是**同一有理式**。守卫 `act_parity_test`（历史测量）：前向一致 max $`1.192\times 10^{-7}`$（1 ULP）、导数对中心差分 max $`1.925\times 10^{-7}`$、门控更新 max $`7.749\times 10^{-7}`$。修复前精确 `tanhf`/`expf` 与 Padé 的偏差约为 $`2.35\times 10^{-2}`$ / $`1.59\times 10^{-1}`$。判据：语义等价 $`\le 4`$ ULP。

`TAO_FAST_ACT=0` 可回到精确激活，此时训练必须用 `-DTAO_TRAIN_EXACT_ACT`，否则再次违反 R4。

---

## 7. 文献：借鉴什么、不借鉴什么

**[1] Vaswani et al., 2017.** *Attention Is All You Need.* NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

自回归 LM、teacher forcing、逐步 CE、残差流、绑定嵌入。**不采用** $`\mathrm{softmax}(QK^{\mathrm{T}}/\sqrt{d_k})V`$ 与 KV cache。序列混合改成定理 1–3 的递归。

**[2] Hochreiter & Schmidhuber, 1997.** *Long Short-Term Memory.* Neural Computation.

门控记忆防止梯度沿时间爆炸/消失。短状态是 GRU 式凸组合，不是 LSTM 的四门细胞。

**[3] Cho et al., 2014.** *Learning Phrase Representations using RNN Encoder–Decoder.* [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)

$`s_t=(1-\sigma)\odot s_{t-1}+\sigma\odot\tanh(u)`$ 与 GRU 更新门同型，状态有界（定理 1）。

**[4] Katharopoulos et al., 2020.** *Transformers are RNNs.* ICML. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)

线性注意力可写成有限状态 RNN。我们走显式矩阵 $`M`$ 的秩一修正，而不是核特征的 $`\phi(K)^{\mathrm{T}}V`$ 累加。

**[5] Schlag, Irie, Schmidhuber, 2021.** *Linear Transformers Are Secretly Fast Weight Programmers.* ICML. [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)

Delta 规则作为快权重：$`W\leftarrow W+\beta(v-W\phi(k))\phi(k)^{\mathrm{T}}`$。定理 2–3 即该更新在本仓库的矩阵记忆上的陈述。`mem.key/query/value` 沿用 FWP 术语，**不是**注意力头。

**[6] Gu & Dao, 2023.** *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

选择性（输入依赖）线性递归作为注意力替代的路线。落地处：$`\sigma(a_t)`$ 与 $`\beta_t(h)`$。Mamba 是对角 SSM；我们不是 —— $`M`$ 是关联记忆，$`s`$ 是 GRU 式短状态。

**[7] Beck et al., 2024.** *xLSTM: Extended Long Short-Term Memory.* [arXiv:2405.04517](https://arxiv.org/abs/2405.04517)

mLSTM 的协方差记忆 $`C`$ 与本仓库 $`M`$ 同型（矩阵状态 + 外积写入）。TFLA（tiled flash linear attention）若迁移，应对准 **$`C\equiv M`$ 的分块**，不能对准 Transformer KV。

**[8] Zhang & Sennrich, 2019.** *Root Mean Square Layer Normalization.* [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

$`R_\gamma`$ 即 RMSNorm，无均值中心化，与残差预归一化一致。

**[9] Sennrich, Haddow, Birch, 2016.** *Neural Machine Translation of Rare Words with Subword Units.* ACL. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

字节级 BPE：先当字节，再 merge，避免 UNK。

**[10] Loshchilov & Hutter, 2019.** *Decoupled Weight Decay Regularization.* ICLR. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

AdamW：衰减加在参数上，不混进自适应梯度。

**[11] Li, Zhang, Liu, 2016.** *Ternary Weight Networks.* [arXiv:1605.04711](https://arxiv.org/abs/1605.04711)

逐行 $`\lbrace -1,0,+1\rbrace`$ 与尺度 $`\alpha`$。定理 5 给出本实现的精确支撑选择，而非阈值启发式。

**[12] Bengio, Léonard, Courville, 2013.** *Estimating or Propagating Gradients Through Stochastic Neurons.* [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)

STE：前向离散，反向当恒等。三值投影的学习通道。

**[13] He et al., 2016.** *Deep Residual Learning for Image Recognition.* CVPR.

层间 $`x\leftarrow x+R(r)`$，让循环模块学修正而不是整幅映射。

**[14] DeepSeek-AI, 2026.** *DeepSeek-V4.1-Flash* 技术报告.

KV cache 是注意力解码的主成本。我们把该论点推到端点：**没有注意力就没有 KV**，状态恒定 257 KiB。**没有**实现 CSA2 / 层次稀疏注意力 / 跨层 KV 复用；那些是注意力家族的压缩术，与定理 3 的地址记忆不是同一对象。

**[15] Qwen Team, 2026.** [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。

当代稠密混合：64 层里 48 层 Gated DeltaNet、16 层满注意力，外加宽 FFN（中间维 17408）与视觉塔。文本侧约 27B，官方 BF16 约 54 GB。记忆更新与 [5][7] 同族，但质量仍靠注意力层和 FFN 撑着。

### 对比：优势与缺点

夭夭 = GRU 式短状态 $`s`$ + FWP/delta 矩阵 $`M`$ + **无 FFN** + 三值权重。下面只比**序列混合与记忆**，不比 AdamW / BPE / RMSNorm。9.5M 的生成质量不能拿来宣称赢过下列已规模化的模型；表里的「优势」首先是渐近与硬件，不是榜单。

| 对照 | 我们相对的优势 | 我们相对的缺点 |
|---|---|---|
| Transformer [1] | 没有 $`QK^{\mathrm{T}}V`$ 与 KV cache；逐步状态 **257 KiB、与上下文长度无关**；本机 CPU 约 $`5\times 10^3`$ tok/s 且不随长度掉。 | 不能对全文做内容检索；当前问句绑不稳（v2 评测串题）。无 FFN，参数几乎都在投影和词表头，同样宽度深度远小于 Transformer 的有效容量。 |
| LSTM [2] / GRU [3] | 在 GRU 式 $`s`$ 之外多一张可寻址的 $`M`$（定理 2–3：按 key 覆写、正交不互扰）。$`s`$ 有界，避免靠发散幅值「记事」。 | $`s`$ 容量仍是 128 维门控；$`M`$ 每步整表读+写，不是 LSTM 那种廉价标量细胞。长程依赖没有注意力那种任意对齐。 |
| 线性注意力 [4] | 显式 $`M`$ 秩一修正，读写分离（$`\hat k`$ 写、$`q`$ 读），不必维护 $`\phi(K)^{\mathrm{T}}V`$ 核特征。融合式把 $`M`$ 扫描从 3 次收成 2 次，线程划分 bitwise 一致。 | 线性注意力常见实现仍带 FFN；我们没有。核方法可以把记忆压进特征空间，我们把容量钉死在 $`m\times d_k`$ 球面可分辨方向上（有限，不是压缩版注意力）。 |
| FWP / delta 规则 [5] | 定理 2–3 就是该更新在本仓库的陈述；`mem.key/query/value` 名副其实，不是注意力头。三值 + CPU VNNI 把逐步算力落到笔记本上。 | 当前实现 **稠密扫 $`M`$**：扩到等效 27B 时 $`M`$ 本身约 6 GB，读+写与权重一个量级。没有 [5] 论文里可学的核 $`\phi`$。 |
| Mamba [6] | $`M`$ 是关联存储，不是对角 SSM；一条地址可覆写而不涂掉正交地址。短状态与记忆分工明确。 | Mamba 有规模化语言模型证据和硬件友好扫描；我们没有对标规模。对角 SSM 每步更轻；$`M`$ 是 $`m\times d_k`$ 稠密矩阵，带宽更凶。 |
| xLSTM [7] | 与 mLSTM 的 $`C`$ 同型，优化应对准 $`C\equiv M`$ 分块，而不是 KV。无注意力、无 KV。 | xLSTM 混合 sLSTM/mLSTM 且带 FFN，有更大训练体量。我们 $`L=2`$、无 FFN，表达宽度差一截。 |
| DeepSeek 长上下文压缩 [14] | 没有 KV 就没有 KV 压缩问题；257 KiB 恒定，不随 32K/262K 涨。 | **没有** CSA2 / 层次稀疏注意力。那些能在注意力模型里买长上下文质量；我们买不到，长程只能靠 $`M`$ 的地址容量。 |
| Qwen3.8-27B [15] | 同类 DeltaNet 记忆上我们走纯循环、无满注意力层。9.5M 三值约 2.5 MB，本机千 tok/s；同等 27B 稠密对齐后逐步仍 O(1)。 | 对方 27B 里大部分是 FFN，还有 16 层满注意力托质量。我们无 FFN、无那 16 层，等效 27B 在本机约 **2 tok/s**（带宽墙），且 9.5M 实测内容弱。 |
| 三值网络 [11] + STE [12] | 盘上 2-bit，解码 int8 VNNI；同样参数个数的工作集比 BF16 小 8 倍。逐行最优支撑（定理 5），不是随意阈值。 | STE 把量化误差当恒等，容量/精度差于同尺寸浮点。零元是数值稀疏，访问仍是稠密 tile，救不了内存墙。 |

**读表时的一条硬约束。** 稠密逐步下，总参数 ≈ 每步碰到的参数。稀疏（热集进 L3、冷集在内存）只在「每步活跃集 ≪ 总参数」时改变上表的速度列；本版未做。词频分片救不了全词表头和每步必用的层矩阵。

---

## 8. 实证（如实，到 2026-09-12）

硬件：AMD Ryzen 9 7940H（Zen 4，AVX-512）+ RTX 4070 Laptop 8 GB。

**v1 过夜课**（`release/L1_qa_cot_v1.dsb`，已冻结、不覆盖）：APE 2.5 万 + alpaca 1.5 万 + 逐位加减，3200 步，末步 NLL 约 2.9–3.5。评测（`tok_qa.bbp`，8 线程，旧 presence 惩罚）：

- 问答 held-out 200：空回复 0%；乱码 1%；**复读 50%**；bigram F1 **0.008**。
- CoT held-out 280：格式（「思考 / 计算 / 答案」）几乎总有；APE 精确 14%；个位加法 2.5%。
- 病根：APE 模板占比过高，9.5M 把「思考：计算…」当成万能回复；算术格式对、得数不对。

**解码修复后**（频率惩罚 + 3-gram + 连写收束，同一 v1 权重）：held-out 复读 **50% → 0%**。内容错误仍在，那是权重问题。

**v2 重训**（`release/L1_qa_cot_v2.dsb`，不覆盖 v1）：alpaca **3.6 万** + 短 APE **8 千** + 五种问法的 1–20 加减；7 片 × 400 = 2800 步；末步 NLL 约 **4.50**。词表仍为 `tok_qa.bbp`。不混 wiki/code。

- 问答 200：空 0%；乱码 3.5%；复读 **0%**；bigram F1 **0.0125**；长度比 0.989。
- CoT 280：空 0；复读 0；含思考链 262/280；APE 精确 **25/200（12.5%）**；个位加减 **3/80（3.8%）**。
- 内容仍弱：日常问答常串题、串模板；CoT 轨迹格式在、算术多数错。alpaca 加重没有把 APE 句式从日常问答里拆干净。

9.5M 的诚实预期：学格式、短回复、短 CoT 外壳；不是应用题能力，也不是开放域知识。

本机 CPU 贪心（v1 权重，8 线程，512 forced）：均值约 **5.08×10³ tok/s**。外推：同等算子、本机 DDR5-5200，约 **5.3×10⁸** 参数可维持 100 tok/s（int8 核、带宽墙）；Qwen3.8-27B 那种 27B 有效文本权重对齐后约 **2 tok/s**。词频把权重拆进 L3 救不了全词表头和每步必用的层矩阵。稀疏分支留到后续，本版不做。

---

## 9. 发布权重

检查点是 DSB2，不是 Hugging Face `safetensors`。与 `build/` 里的训练输出是同一份拷贝；不要覆盖 `build/L1_qa_cot` 或 `build/L1_qa_cot_v2`。

| 文件 | SHA256（前 8 位） | 说明 |
|---|---|---|
| [`release/tok_qa.bbp`](release/tok_qa.bbp) | `13fd68ba` | 问答 BPE，16,123 merges。解码必配。 |
| [`release/L1_qa_cot_v2.dsb`](release/L1_qa_cot_v2.dsb) | `77674df5` | **当前推荐**。2800 步，约 2.5 MB。 |
| [`release/L1_qa_cot_v1.dsb`](release/L1_qa_cot_v1.dsb) | `71febe99` | 冻结对照。APE 偏重，模板劫持明显。 |
| [`release/SHA256SUMS.txt`](release/SHA256SUMS.txt) | | 完整摘要。 |

```bat
scripts\build_h2r_cpu.bat
set TAO_TOKENIZER=%CD%\release\tok_qa.bbp
set TAO_CPU_THREADS=8
build\h2r_cpu.exe release\L1_qa_cot_v2.dsb --max-out 256
```

stdin 一行一问；`/reset` 清循环状态，`/quit` 退出。完整校验见 `release/SHA256SUMS.txt`。

---

## 10. 训练指导

针对 **9.5M、本算子、8 GB 笔记本 GPU**。目标是可复现的问答+短 CoT，不是把夭夭训成 Qwen。

**形状。** 锁 `L=2,d=512,s=128,m=512,dk=64,V=16384`。`e=1024` 只为序列化兼容，本算子无 FFN。不要为了「更像大模型」加层或加宽：4070 Laptop 8 GB 上再叠 wiki/code 或把 $`d`$ 拉到几千，会先爆显存和吞吐，而不是先长能力。

**词表。** 用发布的 `tok_qa.bbp`（digest `13fd68ba…dd75`）。默认冻结；只有设 `TAO_ALLOW_TOKENIZER=1` 才允许换表。问答课不要重训 BPE。检查点与词表必须成对，解码器按 bundle 里的 digest 校验。

**R4。** 训练与解码必须同一激活：默认三阶 Padé（`fast_act`）。一边 `TAO_FAST_ACT=0`、另一边不设 `-DTAO_TRAIN_EXACT_ACT` 会静默把权重训到另一套非线性上。

**目录。** `train_shards` 的 OUT 已存在则失败退出。不要覆盖 `build/L1_qa_cot`、`build/L1_qa_cot_v2` 或 `data/qa_cot*`。新跑开新目录。

**语料配比（已踩过的坑）。**

| | v1（勿再复用当主课） | v2（当前脚本） |
|---|---:|---:|
| alpaca-zh 问答 | 15000 | **36000** |
| APE210K「计算」+「答案」 | 25000 | **8000**（答案截到 360 字） |
| 1–20 逐位加减 | 一种问法 | **五种问法** |
| wiki / code | 不混 | 不混 |

v1 的失败模式是 APE 句式变成万能回复。v2 把日常问答加重之后，复读靠解码器压到 0，但内容仍差：9.5M 记不住应用题，也绑不稳当前问句。下一步若继续训，优先 **更干净的短问答、少模板、多种问法**，不要靠堆 APE 条数。

数据流：对话 jsonl → `build_corpus --format jsonl --dialogue-out`（TLP2 分片）→ `train_shards`。助手位才计损失。建议每片约 6500 篇、每片 400 步。

**优化器（v2 实测）。** AdamW $`\beta_1=0.9,\beta_2=0.999`$，主权重衰减 0.01；`TAO_LR=0.001`；20 步线性 warmup；`TAO_LR_DECAY_START=400`、`TAO_LR_DECAY_STEPS=2800`、`TAO_LR_MIN=1e-5`；`TAO_GRAD_CLIP=1.0`。`TAO_OPT_OFFLOAD=0`，`TAO_OPT_STATE=1`。槽位/宽度 `32 32`。换片后 NLL 跳一下是正常的。

一键（拒绝覆盖已有 OUT）：

```bat
scripts\run_qa_cot_v2.bat
```

手工：

```
train_shards SHARD_DIR TOKENIZER.bbp OUT_DIR STEPS_PER_SHARD [SLOTS WIDTH] [RESUME_DIR]
```

**评测。** 不要看训练 NLL 单独下结论。

```bat
set TAO_TOKENIZER=%CD%\release\tok_qa.bbp
set TAO_CPU_THREADS=8
node scripts\eval_real.mjs release\L1_qa_cot_v2.dsb data\qa_cot_v2\heldout_qa.jsonl 200
node scripts\eval_cot.mjs release\L1_qa_cot_v2.dsb data\qa_cot_v2\heldout_cot.jsonl
```

看空回复、乱码、复读、bigram F1，以及 CoT 是否真算对，而不是只看「思考：」出现没有。

**不要做的。** 把 Transformer 的 KQV / KV cache / CSA2 / 层次稀疏注意力接到本算子上。`mem.key/query/value` 是关联记忆的写方向、读方向、载荷。稀疏（按命中率把权重拆进 L3）是后续研究，本版没有实现。

---

## 11. 构建与入口

工具链：MSVC + nvcc（`sm_89`），C++17。

| 脚本 | 产物 |
|---|---|
| `scripts/build_train_shards.bat` | `build/train_shards.exe` |
| `scripts/build_h2r_cpu.bat` | `build/h2r_cpu.exe`（`/arch:AVX512`） |
| `scripts/build_bench_decode.bat` | `build/bench_decode_tps.exe` |
| `scripts/build_corpus_tools.bat` | `build/build_corpus.exe` 等 |
| `scripts/run_qa_cot_v2.bat` | v2 问答+CoT 训练（拒绝覆盖已有 OUT） |

```
h2r_cpu MODEL.dsb [--rep-pen F] [--max-out N]
```

环境变量（节选）：`TAO_TOKENIZER`、`TAO_ALLOW_TOKENIZER`、`TAO_CPU_THREADS`、`TAO_VNNI`、`TAO_FAST_ACT`、`TAO_LR`、`TAO_LR_DECAY_START`、`TAO_LR_DECAY_STEPS`、`TAO_GRAD_CLIP`、`TAO_REP_PEN`、`TAO_MAX_OUT`。

`data/` 与 `build/` 默认 gitignore；`release/*.dsb` 与 `release/tok_qa.bbp` 入库。

GitHub：`main` 为本 0.1.1 算子。此前远程上的 0.2 量级探索线保留在分支 [`legacy`](https://github.com/TaoLe-si/Yaoyao/tree/legacy)。

---

## 12. 版本

- **0.1.1** — `dual-state-4-noffn-delta-mem-input-sqrt-d`：delta 矩阵记忆取代向量记忆；$`L=8\to 2`$；Padé 激活统一（R4）；CPU 头 VNNI / `vnni8` / 缓冲复用；固定 $`s|M`$ 对齐进 L2 + 词表头 NTA；生成端频率重复惩罚（无窗口）；发布 v1/v2 检查点。
- **0.1.0** — `dual-state-3-noffn-input-sqrt-d`：门控向量双状态，无 FFN。
