# Yaoyao / 夭夭（TaoVm）

原生 C++/CUDA **三值双状态语言模型**：GPU 上训练有效三值权重，CPU 上用同一前向方程做推理。本仓库描述的是当前正式主干（arch2 dual-state），不是旧版 D256 / 可逆链服务。

**明确不做的宣称：** 不等于 Transformer；没有无限精确记忆；覆盖式状态更新不可逆；验证 NLL 下降不等于对话质量；CPU 不做神经网络训练。

---

## 1. 仓库布局

| 路径 | 内容 |
|---|---|
| `src/` | 全部 C/C++/CUDA 头与实现（模型、训练、CPU 解码、测试） |
| `docs/` | 历史决策、诊断、试验记录 |
| `scripts/` | 构建、导出、验收、分析脚本 |
| `build/` | 本地产物与检查点（不入库） |
| `experiments/` | 本地试验资产（大文件不入库） |

训练与推理的**数学定义**以 `src/dual_state_cpu.hpp`、`src/dual_state_autograd.cuh`、`src/dual_projection_sorted.cuh`、`src/dual_state_sorted_trainer.cuh` 为准。CPU 前向与 GPU 训练图必须实现同一组方程。

编译时把 `src/` 加到 include 路径（`/I src` 或 `-I src`）。头文件互相用引号包含，不带目录前缀。

---

## 2. 规模与张量清单

配置 `tao::dual::Config`（`src/dual_state_config.hpp`）：

$$
L=8,\quad d=512,\quad s=128,\quad m=512,\quad e=1024,\quad V=16384.
$$
| 符号 | 值 | 含义 |
|---|---:|---|
| $L$ | 8 | 层数 |
| $d$ | 512 | 残差 / 读出宽度 |
| $s$ | 128 | 每层局部状态 $s$ |
| $m$ | 512 | 每层长期状态 $m$ |
| $e$ | 1024 | 前馈扩张（SiLU） |
| $V$ | 16384 | 词表 |

参数总量（按 schema 精确计数）：

- 三值矩阵 **113** 个，元素 **30,146,560**
- 浮点张量 **58** 个（bias / RMS $\gamma$），元素 **39,424**
- 合计 **30,185,984**（约 30.19M）

每个 token 的循环状态是固定维

$$
\mathrm{state}=\bigl(s^{(\ell)},m^{(\ell)}\bigr)_{\ell=1}^{L}\in\mathbb{R}^{L(s+m)}=\mathbb{R}^{5120},
$$
**不随序列长度增长**，也没有 KV cache。新文档把状态清零。这只说明状态占用 $O(1)\) 于长度，不说明容量或可逆记忆。

### 2.1 张量 schema

记层前缀 $\ell$。三值矩阵用行尺度 $\alpha$ 与 $q\in\{-1,0,1\}^{\mathrm{row}}$ 表示，有效行 $\alpha q$。浮点张量不量化。

**全局**

| 名称 | 形状 | 三值 |
|---|---|---|
| `embedding` | $V\times d$ | 是（词嵌入，解嵌共用） |
| `vocab.bias` | $V$ | 否 |
| `final.norm` | $d$ | 否（最终 RMS $\gamma$） |

**每一层 $\ell=0\ldots L-1$**

| 名称 | 形状 | 三值 | 角色 |
|---|---|---|---|
| `s.candidate.{x,s}` | $s\times d,\; s\times s$ | 是 | 局部候选 |
| `s.candidate.bias` | $s$ | 否 | |
| `s.gate.{x,s}` | $s\times d,\; s\times s$ | 是 | 局部门 |
| `s.gate.bias` | $s$ | 否 | |
| `m.candidate.{x,s,m}` | $m\times d,\; m\times s,\; m\times m$ | 是 | 长期候选 |
| `m.candidate.bias` | $m$ | 否 | |
| `m.gate.{x,s,m}` | 同上 | 是 | 长期门 |
| `m.gate.bias` | $m$ | 否 | |
| `read.s`, `read.m` | $d\times s,\; d\times m$ | 是 | 状态读出 |
| `ff.up`, `ff.down` | $e\times d,\; d\times e$ | 是 | SiLU 前馈 |
| `input.norm`, `read.norm`, `ff.norm` | $d$ | 否 | 三处 RMS $\gamma$ |

局部分支 **不读** $m$。长期分支读 **更新后的** $s$ 与旧 $m$。层输出残差进入下一层，不再写回本层当前 token 的 $(s,m)$。

---

## 3. 词表、BPE 与数据契约

### 3.1 特殊 ID

`src/language_data_contract.hpp`：

$$
\begin{aligned}
\mathrm{BOS}&=256,\quad
\mathrm{USER}=257,\quad
\mathrm{ASSISTANT}=258,\\
\mathrm{TURN\_END}&=259,\quad
\mathrm{EOS}=260.
\end{aligned}
$$
- $0\ldots255$：原始字节
- $261\ldots V-1$：BPE 合并符号。合并条数 $\le 16123$，故 $V=256+5+16123=16384$

文件格式 **BBP1**（`src/tokenizer_file.hpp`）：魔数 `BBP1` + little-endian $n$ + $n$ 对 $(a,b)\) 合并。正式分词器 SHA-256：

`34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333`

编码：先把 UTF-8 当字节 ID，再按合并表从早到晚做非重叠替换。解码：把 $\ge 261$ 的 ID 按合并表展开回字节。拟合只在消息体上做，不跨消息。

### 3.2 TLP2 对话流

当前训练/验证用 **TLP2**（`src/bpe_pilot_reader.hpp`），**不含 EOS**。记录：

1. 4 字节魔数 `TLP2` + 64 字节分词器 SHA 的 ASCII
2. 若干文档：`uint32` 长度 $n$，随后 $n$ 个 `(id_lo, id_hi, loss)`

文法：`BOS`（不监督）后重复

$$
\mathrm{ROLE}\,\mathrm{BODY}^*\,\mathrm{TURN\_END},
$$
`ROLE` 为 USER 或 ASSISTANT。正文 ID 满足 $t<256$ 或 $t\ge 261$。`loss=1` 当且仅当该 token 属于助手正文或助手的 `TURN_END`。用户侧全部 $\mathrm{loss}=0$。

形式化：令序列 $(x_t,\lambda_t)_{t=0}^{T}$，预测目标是 $x_{t+1}$，监督指示 $\lambda_{t+1}\in\{0,1\}$。损失只在 $\lambda_{t+1}=1$ 上累计。

固定验证集：23 篇、4906 个监督位置、6390 个预测位置，数据集 SHA

`69900a8acbb28b09296b13a08c76657aa6261c129e929d59bc83ad9d2f1447b0`

质量口径是助手目标验证 **NLL $\le 2.5$** 且可读对话。训练损失不能替代该口径。

---

## 4. 前向：双状态一层

以下对单层、单 token。向量按元素运算。$\sigma$ 为 logistic，实现为对正负分支稳定的

$$
\sigma(z)=\begin{cases}
(1+e^{-z})^{-1} & z\ge 0\\
e^{z}(1+e^{z})^{-1} & z<0.
\end{cases}
$$
### 4.1 嵌入

令 $E\in\mathbb{R}^{V\times d}$ 为 `embedding` 的**有效**行。arch2 定义 `TAO_INPUT_SCALE`：

$$
x \leftarrow \sqrt{d}\, E_{x_t}\in\mathbb{R}^{d}.
$$
无该宏时尺度为 1。训练、固定验证、CPU 解码必须一致。

### 4.2 RMSNorm

对 $u\in\mathbb{R}^{n}$、$\gamma\in\mathbb{R}^{n}$，$\varepsilon=10^{-5}$：

$$
\mathrm{RMS}(u)=\sqrt{\frac{1}{n}\sum_{j=1}^{n}u_j^{2}+\varepsilon},\qquad
\mathrm{Norm}(u;\gamma)=\gamma\odot \frac{u}{\mathrm{RMS}(u)}.
$$
反向（`src/dual_state_cuda_backward.cuh`）：令 $\mathrm{inv}=1/\mathrm{RMS}(u)$，$s=\sum_i u_i^{2}$，$\delta=(\partial L/\partial y)\odot\gamma$ 与 $u$ 的点积为 $c=\sum_i (\partial L/\partial y_i)\gamma_i u_i$，

$$
\frac{\partial L}{\partial u_i}
= \frac{\partial L}{\partial y_i}\gamma_i\,\mathrm{inv}
- u_i\, c\, \mathrm{inv}^{3}/n,
\qquad
\frac{\partial L}{\partial \gamma_i}
= \frac{\partial L}{\partial y_i} u_i\,\mathrm{inv}.
$$
这是标准 RMSNorm 梯度，不含均值中心化（不是 LayerNorm）。

### 4.3 局部状态 $s$

输入先归一化 $x_n=\mathrm{Norm}(x;\gamma_{\mathrm{input}})$。旧状态 $s_0,m_0$。

$$
\begin{aligned}
u_s &= W_{s}^{x} x_n + U_{s}^{s} s_0 + b_{s}^{u},\\
a   &= W_{a}^{x} x_n + U_{a}^{s} s_0 + b_{s}^{a},\\
s   &= s_0 + \sigma(a)\odot\bigl(\tanh(u_s)-s_0\bigr).
\end{aligned}
$$
逐坐标即凸组合

$$
s_j = \bigl(1-\sigma(a_j)\bigr)s_{0,j} + \sigma(a_j)\tanh((u_s)_j).
$$
因此若 $|s_{0,j}|\le 1$ 且 $\tanh\in(-1,1)$、$\sigma\in(0,1)$，则 $|s_j|<1$ 在开区间上保持。**门永不严格为 0**，重复的小更新仍可缓慢改写 $s$。

### 4.4 长期状态 $m$

长期分支看到**新** $s$ 与旧 $m_0$：

$$
\begin{aligned}
u_m &= W_{m}^{x} x_n + U_{m}^{s} s + V_{m}^{m} m_0 + b_{m}^{u},\\
g   &= W_{g}^{x} x_n + U_{g}^{s} s + V_{g}^{m} m_0 + b_{m}^{g},\\
m   &= m_0 + \sigma(g)\odot\bigl(\tanh(u_m)-m_0\bigr).
\end{aligned}
$$
同样是凸组合，同样没有硬跳过。初始化把 `m.gate.bias` 设为 $-2$，使 $\sigma(-2)\approx 0.119$，开训时 $m$ 更新偏慢，不是遗忘免疫。

### 4.5 读出、前馈、残差

$$
\begin{aligned}
r &= P_s s + P_m m,\\
x &\leftarrow x + \mathrm{Norm}(r;\gamma_{\mathrm{read}}),\\
h &= W_{\uparrow}\,\mathrm{Norm}(x;\gamma_{\mathrm{ff}}),\\
h &\leftarrow h\odot\sigma(h) \quad (\mathrm{SiLU}),\\
x &\leftarrow x + W_{\downarrow} h.
\end{aligned}
$$
SiLU 导数（实现与公式一致）：

$$
\frac{d}{dz}\bigl(z\sigma(z)\bigr)=\sigma(z)+z\sigma(z)\bigl(1-\sigma(z)\bigr).
$$
### 4.6 状态更新的反向

令 $y=s_0+\sigma(g)\odot(\tanh(u)-s_0)$（$s$ 与 $m$ 同形）。实现保留 **旧状态、tanh 前候选、sigmoid 前门**：

$$
\begin{aligned}
\frac{\partial L}{\partial s_0}
&= \frac{\partial L}{\partial y}\odot\bigl(1-\sigma(g)\bigr),\\
\frac{\partial L}{\partial u}
&= \frac{\partial L}{\partial y}\odot\sigma(g)\odot\bigl(1-\tanh^{2}(u)\bigr),\\
\frac{\partial L}{\partial g}
&= \frac{\partial L}{\partial y}\odot\bigl(\tanh(u)-s_0\bigr)\odot\sigma(g)\odot\bigl(1-\sigma(g)\bigr).
\end{aligned}
$$
见 `ds_update_backward`。

### 4.7 词表解嵌

最终

$$
z = E\,\mathrm{Norm}(x;\gamma_{\mathrm{final}}) + b \in\mathbb{R}^{V}.
$$
`linear("embedding", ·, V)` 对 $E$ 的每一行做点积，即 **tied embedding**：第 $i$ 个 logit 是有效词向量 $E_i$ 与归一化残差的内积。

### 4.8 层间数据流（一块不漏）

对 token $x_t$，层 $\ell=0\ldots L-1$ 顺序：

1. $x_n\leftarrow\mathrm{Norm}(x;\gamma^{(\ell)}_{\mathrm{input}})$
2. 用 $(x_n,s^{(\ell)})$ 写 $s^{(\ell)}$
3. 用 $(x_n,s^{(\ell)}_{\mathrm{new}},m^{(\ell)})$ 写 $m^{(\ell)}$
4. 读出加残差
5. SiLU 前馈加残差
6. $x$ 进入 $\ell+1$

没有注意力、没有位置编码、没有因果 mask。时间依赖只通过 $(s,m)$ 跨 token 传递，以及 TBPTT 在 256 步边界对计算图的切断。

---

## 5. 三值有效权重

### 5.1 主权重与有效权重

每个三值矩阵有 FP32 **主权重** $W$（Adam 更新对象）和 **有效权重** $\widetilde W$（前向/反向 matvec 实际使用）。浮点张量两者相同，拷贝主权重。

行 $r$ 独立投影。令 $w\in\mathbb{R}^{c}$ 为该行，$c\in\{d,s,m,e,V\}$ 且 $c\le 1024$。

### 5.2 行投影（sorted abs / 前缀 $k$）

对固定行，在

$$
\widetilde w = \alpha q,\qquad q\in\{-1,0,1\}^{c},\quad \alpha>0
$$
中选使 $\|w-\alpha q\|_2^{2}$ 最小的 **幅值前缀支撑** 解。

**命题。** 若约束 $\mathrm{supp}(q)=S$ 且 $q_j=\mathrm{sign}(w_j)$（$w_j=0$ 则 $q_j=0$），则最优尺度为

$$
\alpha_S^{\star}=\frac{1}{|S|}\sum_{j\in S}|w_j|
$$
（$S=\emptyset$ 或全零时实现取 $\alpha=1$，有效行全 0）。代入后

$$
\|w-\alpha_S^{\star} q\|_2^{2}
= \|w\|_2^{2} - \frac{1}{|S|}\Bigl(\sum_{j\in S}|w_j|\Bigr)^{2}.
$$
因此最小化重构误差 $\Leftrightarrow$ 最大化

$$
J(S)=\frac{1}{|S|}\Bigl(\sum_{j\in S}|w_j|\Bigr)^{2}.
$$
**命题。** 在 $|S|=k$ 固定时，最优 $S$ 是 $|w_j|$ 最大的 $k$ 个坐标（并列按列下标升序，与 bitonic 排序稳定规则一致）。

证明：$J$ 对 $|w_j|$ 单调；把较小幅值换入集合不会增大和的平方。

实现（`ds_project_sorted`）对 $|w|$ 降序、下标升序做 bitonic 排序，扫描前缀

$$
J_k=\frac{1}{k}\Bigl(\sum_{i=1}^{k}|w|_{(i)}\Bigr)^{2},
$$
取最大 $J_k$ 的 $k^{\star}$，$\alpha=k^{\star-1}\sum_{i=1}^{k^{\star}}|w|_{(i)}$，前 $k^{\star}$ 个非零项写 $\mathrm{copysign}(\alpha,w_j)$，其余为 0。零元素即使排进前缀也保持 0。

这是 **行共享 $|\alpha|$** 的三值，不是逐元素独立量化，也不是 2-bit 存储参与训练算术。磁盘 DSM1/DSB2 才按 2 bit/元素打包：`00=0, 01=+1, 10=-1, 11` 非法。

### 5.3 STE（直通估计）

投影不可微。训练用 **恒等 STE**：前向用 $\widetilde W$，反向把 $\partial L/\partial \widetilde W$ 原样累加到主权重的梯度缓冲，**不**把导数穿过 $k^{\star}$ 或 $\alpha$。

这是有偏估计：真正的 $\mathrm{d}\widetilde W/\mathrm{d}W$ 是分段常值，几乎处处为 0。STE 只提供下降方向，不保证投影后的驻点。

线性层 $y=\widetilde W x$：

$$
\frac{\partial L}{\partial x}=\widetilde W^{\top}\frac{\partial L}{\partial y},
\qquad
\frac{\partial L}{\partial \widetilde W}=\frac{\partial L}{\partial y} x^{\top},
$$
实现为行主序 matvec / 外积累加（`ds_linear_dx`, `ds_linear_dw`）。

---

## 6. 损失

单位置交叉熵，稳定 log-sum-exp。令 $z\in\mathbb{R}^{V}$ 为 logits，目标 $y$：

$$
\ell(z,y)=\log\sum_{j=1}^{V}e^{z_j-m}+m-z_y,
\qquad m=\max_j z_j.
$$
$$
\frac{\partial \ell}{\partial z_j}=\mathrm{softmax}(z)_j-\mathbf{1}_{j=y}.
$$
文档级验证 NLL：

$$
\mathrm{NLL}=\frac{\sum_{t:\lambda_t=1}\ell(z^{(t-1)},x_t)}{\sum_{t:\lambda_t=1}1}.
$$
固定验证用双精度 log-sum-exp、无梯度、不跑 Adam/投影（`src/gpu_fixed_validation_arch2.cuh`）。训练图上的 CE 是 FP32 参考核。无监督位置不写 logit 梯度。

---

## 7. 优化器

AdamW 作用在**主权重**上（`ds_adamw` / `ds_adamw_device`）：

$$
\beta_1=0.9,\quad \beta_2=0.999,\quad \varepsilon=10^{-8}.
$$
一步有 $N$ 个监督 token。先对全部参数拼接的梯度做均值 $g\leftarrow G/N$，再做全局 $\ell_2$ 裁剪：

$$
\mathrm{factor}=\frac{1}{N\,\max(1,\|g\|_2)},
\qquad
\tilde g = G\cdot \mathrm{factor}.
$$
即：先按监督 token 平均，若平均梯度范数 $>1$ 则缩放到 1，否则不放大。

偏置修正用 **更新后的** 步号 $t$：

$$
\hat m_t=\frac{m_t}{1-\beta_1^{t}},\quad
\hat v_t=\frac{v_t}{1-\beta_2^{t}}.
$$
$$
W \leftarrow W - \eta\Bigl(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}+\lambda W\Bigr).
$$
$\lambda=0.01$ **仅三值矩阵**；bias / RMS $\gamma$ 的 $\lambda=0$。默认生产学习率由 control 文件给出；独立修复试验用过 $\eta=2.5\times 10^{-4}$ 与 $\eta=10^{-4}$。

每步结束：Adam → `project()` 刷新有效权重 → `zero_grad`。

---

## 8. 截断 BPTT、槽位与图

- 图：`4` 个序列槽 $\times$ `8` 次 accum $\times$ 宽 `256`
- 每 256 个时间步 `detach()`：拷贝 $(s,m)$ 的值、丢掉 tape，梯度仍留在参数节点上（截断 BPTT）
- `SequenceSlots`：槽位串行共享参数/梯度，状态独立；新文档重置 $(s,m)$
- 一步最多 $4\times 8\times 256$ 个位置，实际监督数因文档边界和 mask 更少

这解释了每步 `targets` 大约 3k–5k、且随文档长度波动。

CUDA Graph（`ReusableBatchGraph`）捕获「256 步前向 + 反向」，每步只上传 batch 计划并回写槽位状态。验证必须在 `slots.active==-1$ 且优化器边界调用。

---

## 9. 初始化（seed 713）

`src/dual_state_initialization.hpp`，`std::mt19937`。

浮点：

- 普通 bias：0
- `*.norm`：1，但 `read.norm` 为 $1/\sqrt{2L}$
- `m.gate.bias`：$-2$

三值主权重 $\mathcal{N}(0,1/\mathrm{fan})$：

| 张量 | fan |
|---|---|
| embedding, ff.up | $d$ |
| ff.down | $2 L e$ |
| read.s / read.m | $s+m$ |
| s.candidate / s.gate | $d+s$ |
| m.candidate / m.gate | $d+s+m$ |

Fan-in 选择对应残差方差的启发式，不是定理。初始化后立刻投影得到第一份有效三值矩阵。

---

## 10. 磁盘格式

- **DSM1**：裸模型。三值行：`float32 α` + 每 4 个三值 1 字节（低位先）。非法码 `11` 拒绝。
- **DSB2**：推理包。`DSB2` + 清单长度 + 载荷长度 + FNV-1a 64 校验 + 清单（算子 ID、分词器 SHA、schema）+ DSM1 载荷。arch2 算子 ID 为 `dual-state-2-input-sqrt-d`。拒绝覆盖已有文件。
- **SCP**：训练槽位检查点（主权重、Adam $m,v$、步号、游标、$(s,m)$）。与 DSB 一起写时不得覆盖已有 stem。

CPU 推理加载 DSB 中的**有效**权重，执行与 `CpuModel::step` 相同的方程。CPU 路径不做 Adam、不做投影、不做反向。

---

## 11. 训练 / 推理分工

| | GPU | CPU |
|---|---|---|
| 神经网络训练 | 是 | **否** |
| 前向 | 训练图 / 固定验证 | 解码 |
| 数据导出、过滤、SHA | 否 | 允许 |
| 优化器 | AdamW + 投影 | — |

正式训练入口：`src/train_yaoyao_graph_gpuval.cu`（`STOP_TRAINING` 存在则只恢复、不更新）。独立修复试验写到独立目录，禁止覆盖 `yaoyao_graph_step_1200/1204`。

---

## 12. 能力边界（与方程一致）

1. **不是注意力。** 无 $QK^{\top}$，无长度正比的 cache。长程依赖只通过固定维 $(s,m)$ 的门控覆盖。
2. **覆盖不可逆。** $s\leftarrow (1-a)s+a\,u$ 在 $a\ne 0$ 时丢掉旧坐标的信息。不能从当前状态重建任意历史 token。
3. **门不是硬选择。** $\sigma$ 值域 $(0,1)$。要严格跳过必须另定义两端相同的硬门与梯度，禁止只在推理阈值化。
4. **三值是行共享尺度的离散可行集。** STE + 投影使有效权重跳跃；训练 NLL 与验证 NLL 可以脱节。
5. **验证口径冻结。** 23 篇 / 4906 监督 / 上述两个 SHA。换训练集不能动验证集。
6. **质量。** 目标仍是助手验证 NLL $\le 2.5$ 且可读。更低的训练或验证 NLL 单独都不构成成功。

---

## 13. 构建

Windows / MSVC x64，CUDA sm_89（RTX 4070 Laptop）。CPU 测试：

```bat
scripts\build_project.bat
```

GPU 训练示例（需 vcvars + nvcc，且 `src/` 在 include 路径中）：

```bat
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -I src src\train_yaoyao_graph_gpuval.cu -o build\train_yaoyao_graph_gpuval.exe
```

Python 仅用于本地 OpenAI 兼容封装（`requirements.txt`），**不参与训练**。

---

## 14. 历史

`docs/` 中有旧 D256、可逆链、门控检索等试验记录。那些文件描述的是已经不做的路径；当前可训练/可解码的定义是本文第 2–10 节。不要把旧资产的能力数字套到 dual-state 主干上。
