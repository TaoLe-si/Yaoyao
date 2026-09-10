# 夭夭 Yaoyao 0.1

夭夭是原生 C++ / CUDA 三值双状态语言模型。正式实现版本为 **0.1.0**：GPU 训练，CPU 常驻权重贪心解码；每层保留短状态和记忆状态，**移除整个 FFN 升维／降维分支**。

0.1 是架构与实现版本。当前模型从零训练中，尚未验证成熟的自然语言对话能力。step 100 的九次对话测试全部立即输出结束标记；step 500与600均在同一组九次问答中循环输出“好的”到64-token截断，有效回答0/9；验证 NLL 下降不能替代生成质量评估。

## 架构

| 项目 | 0.1 配置 |
|---|---:|
| 层数 | 8 |
| 主通道 d | 512 |
| 每层短状态 s | 128 |
| 每层记忆状态 m | 512 |
| BPE 词表 | 16,384 |
| 三值矩阵 | 97 个，21,757,952 个元素 |
| 浮点偏置／归一化参数 | 50 个张量，35,328 个元素 |
| 合计参数 | **21,793,280** |
| 每会话循环状态 | 5,120 个 float32，20 KiB |
| 权重算子身份 | `dual-state-3-noffn-input-sqrt-d` |

词嵌入与输出头共享矩阵。矩阵有效权重为逐行 `alpha * q`，其中 `q ∈ {-1,0,1}`；训练保存浮点 master weights 和 Adam 矩。偏置、RMSNorm 参数保持浮点。没有随上下文长度增长的 KV cache；固定状态不代表无限记忆或可逆计算。

每层计算流程：

```text
token embedding × sqrt(d)
  → RMSNorm(input)
  → 更新 s：读取当前输入与旧 s
  → 更新 m：读取当前输入、更新后的 s 与旧 m
  → read.s(s) + read.m(m)
  → RMSNorm(read) + 输入残差
  → 下一层
最终 RMSNorm → 共享输出头 → vocab.bias → greedy argmax
```

两个状态采用 `state += sigmoid(gate) * (tanh(candidate) - state)`。0.1 不计算或分配 `ff.up`、`ff.down`、`ff.norm`。官方构建启用 `TAO_NO_FFN` 与输入缩放；共享源文件仍保留少量历史条件分支。配置字段 `e=1024` 为现有序列化布局兼容字段，在0.1没有 FFN 计算含义。

CPU、串行 CUDA、批处理 CUDA、固定 GPU 验证均须遵守同一架构。新 DSB/SCP 的 schema 和算子身份与旧 FFN 模型不同；禁止混用旧解码器或直接续用旧 checkpoint。

## 完整数学定义

公式对应正式无FFN构建；矩阵为输出维×输入维。t为token位置，k为优化器步数。每层参数独立，以下省略层上标。

### 输入与状态更新

共享嵌入 E 的形状为 V×d。输入 token z_t 对应：

$$x_t^{(0)}=\sqrt d\,E_{z_t,:}^{\mathsf T}.$$

RMSNorm不减均值、不加偏置，epsilon=10^{-5}：

$$R_\gamma(x)=\gamma\odot x\rho,\qquad \rho=(\|x\|_2^2/n+10^{-5})^{-1/2}.$$

每层先令 h=R_input(x)，更新短状态：

$$u_s=W_{sx}h+W_{ss}s_{t-1}+b_s,$$

$$a_s=G_{sx}h+G_{ss}s_{t-1}+b_{gs},$$

$$s_t=(1-\sigma(a_s))\odot s_{t-1}+\sigma(a_s)\odot\tanh(u_s).$$

记忆候选与门读取更新后的 s 和旧 m，使用独立权重：

$$u_m=W_{mx}h+W_{ms}s_t+W_{mm}m_{t-1}+b_m,$$

$$a_m=G_{mx}h+G_{ms}s_t+G_{mm}m_{t-1}+b_{gm},$$

$$m_t=(1-\sigma(a_m))\odot m_{t-1}+\sigma(a_m)\odot\tanh(u_m).$$

状态读出后加回输入，不经过FFN：

$$r_t=W_{rs}s_t+W_{rm}m_t,\qquad x_t^{(\ell+1)}=x_t^{(\ell)}+R_{\gamma_{read}}(r_t).$$

最终输出共享嵌入矩阵：

$$o_t=E R_{\gamma_f}(x_t^{(L)})+b_v,\qquad p(z_{t+1}=j\mid z_{\le t})=\frac{e^{o_{t,j}}}{\sum_i e^{o_{t,i}}}.$$

CPU贪心选择argmax，无需显式softmax；预测token在下一次前向才被消费。新文档状态归零，会话内保留状态。

### 张量与参数计数

| 每层参数 | 形状 | 数量 |
|---|---|---:|
| 短状态候选/门输入矩阵 | s×d | 2 |
| 短状态候选/门循环矩阵 | s×s | 2 |
| 短状态偏置 | s | 2 |
| 记忆候选/门输入矩阵 | m×d | 2 |
| 记忆候选/门短状态矩阵 | m×s | 2 |
| 记忆候选/门循环矩阵 | m×m | 2 |
| 记忆偏置 | m | 2 |
| read.s / read.m | d×s / d×m | 各1 |
| input/read RMS增益 | d | 2 |

加上全局嵌入 V×d、输出偏置 V 和最终RMS增益 d：

$$P=Vd+V+d+L[2(sd+s^2)+2(md+ms+m^2)+ds+dm+2s+2m+2d]=21{,}793{,}280.$$

行尺度由master投影得出，不是独立Adam参数。2-bit符号存储不代表所有计算或运行内存均为2-bit。训练仍保存浮点master、梯度、Adam矩和激活。

### 三值投影推导

对master的一行 w，求逐行最小二乘重构：

$$\min_{\alpha\ge0,\ q\in\{-1,0,1\}^n}\|w-\alpha q\|_2^2.$$

固定非零集合 S 后，最优符号与对应权重同号。展开误差：

$$J(\alpha,S)=\|w\|_2^2-2\alpha\sum_{i\in S}|w_i|+|S|\alpha^2.$$

令尺度导数为零：

$$
\alpha_{S}^{\ast}=\frac{\sum_{i\in S}|w_i|}{|S|},\qquad J^{\ast}(S)=\lVert w\rVert_{2}^{2}-\frac{\left(\sum_{i\in S}|w_i|\right)^{2}}{|S|}.
$$

固定集合大小 $k$ 时，取绝对值最大的 $k$ 项。排序 $a_1\ge\cdots\ge a_n$，记前缀和 $A_k$：

$$
A_k=\sum_{i=1}^{k}a_i,\qquad k^{\ast}=\operatorname{arg\,max}_{1\le k\le n}\frac{A_k^{2}}{k},\qquad \alpha^{\ast}=\frac{A_{k^{\ast}}}{k^{\ast}}.
$$

实现按绝对值降序、列索引升序排序；分数相同保留较小k。零权重仍为零；全零行尺度设1，有效权重仍为零。这是实数算术下的权重重构最优解，CUDA实现使用浮点；不保证任务损失最优。

离散投影使用identity STE：

$$\frac{\partial\mathcal L}{\partial w^{master}}\approx\frac{\partial\mathcal L}{\partial w^{eff}}.$$

这是替代梯度，不是投影的真实导数；没有独立学习尺度的Adam分支。

### 损失、梯度与TBPTT

目标监督掩码为 mu_t，N为一次更新内实际监督目标数：

$$N=\sum_t\mu_t,\qquad \mathcal L=-\frac1N\sum_t\mu_t\log p(z_{t+1}\mid z_{\le t}).$$

$$\frac{\partial\mathcal L}{\partial o_{t,j}}=\frac{\mu_t}{N}(p_{t,j}-\mathbf1[j=z_{t+1}]).$$

实现先累积未归一化梯度，再按N归一化。训练/验证使用真实历史token，生成使用自己的输出，因此低NLL不保证不循环。

线性层 y=Wx，上游梯度g：

$$\nabla_x\mathcal L=W^{\mathsf T}g,\qquad \nabla_W\mathcal L=gx^{\mathsf T}.$$

门控更新 y=(1-a)⊙v+a⊙c，其中 a=sigmoid(g_0)、c=tanh(u)，上游梯度delta：

$$\delta_v=\delta\odot(1-a),\quad \delta_u=\delta\odot a\odot(1-c^2),\quad \delta_{g_0}=\delta\odot(c-v)\odot a\odot(1-a).$$

第一项只是旧状态的直接路径，候选和门还依赖旧状态，自动微分累加这些间接路径。不能把1-a当作完整循环Jacobian。

RMSNorm的反向：

$$\nabla_x\mathcal L=\rho(g\odot\gamma)-\frac{\rho^3}{n}x\sum_i g_i\gamma_i x_i,\qquad \nabla_\gamma\mathcal L=g\odot x\rho.$$

每256步detach反向路径，保留边界状态数值。每更新累积4槽位×8次图重放；8192只是名义位置容量，实际有效位置和监督目标较少。状态跨块延续不代表梯度贯穿全部历史。

### 优化器与初始化

累积梯度G，归一化g=G/N，范数裁剪：

$$\bar g=g/\max(1,\|g\|_2).$$

用M、U表示Adam一阶二阶矩，区别于模型记忆m：

$$M_k=.9M_{k-1}+.1\bar g,\qquad U_k=.999U_{k-1}+.001\bar g^2,$$

$$\widehat M_k=M_k/(1-.9^k),\qquad\widehat U_k=U_k/(1-.999^k),$$

$$w_k=w_{k-1}-\eta_k\left(\frac{\widehat M_k}{\sqrt{\widehat U_k}+10^{-8}}+\lambda w_{k-1}\right).$$

矩阵lambda=.01，偏置/RMS增益lambda=0。更新master后重新投影，清空梯度。

更新前步数j=0..1999：前100步eta=0.00025(j+1)/100，100..1499保持0.00025，1500之后：

$$\eta(j)=5\times10^{-5}+10^{-4}[1+\cos(\pi(j-1500)/500)].$$

最后一步使用j=1999，略高于理论下限。初始化矩阵服从零均值高斯，标准差1/sqrt(f)：嵌入f=d；短状态分支f=d+s；记忆分支f=d+s+m；读出f=s+m。input/final RMS增益1，read增益1/sqrt(2L)；记忆门偏置-2，其他偏置0。随后立即三值投影，Adam和状态从零开始。

### 数学性质与实现范围

在实数算术、零初始状态下，门控更新是旧状态与tanh候选的逐元素凸组合，所以循环状态在[-1,1]内。残差流不受该界约束；这不能证明梯度稳定、无限记忆、可逆计算或任务收敛。有限精度行为需实测。

持久会话状态复杂度O(L(s+m))。每token稠密计算复杂度：

$$O(Vd+L(sd+s^2+md+ms+m^2+ds+dm)).$$

CPU预填充advance省略输出头，但仍逐token更新循环状态。训练激活内存还随槽位、TBPTT长度和中间节点增加，20KiB状态不是训练显存。

源码依据：[CPU前向](src/dual_state_cpu.hpp)、[自动微分](src/dual_state_autograd.cuh)、[批处理](src/batch_train_graph.cuh)、[三值投影](src/dual_projection_sorted.cuh)、[反向算子](src/dual_state_cuda_backward.cuh)、[优化器](src/dual_state_sorted_trainer.cuh)、[初始化](src/dual_state_initialization.hpp)。这些推导对应正式构建路径；历史CudaResident辅助实现没有完整0.1改造，不能当作正式推理入口。

## 目录

| 路径 | 用途 |
|---|---|
| `src/` | 0.1 依赖闭包及保留的验证／导出入口 |
| `scripts/build_v01.bat` | 构建正式 CPU 与 CUDA 可执行文件 |
| `scripts/train_v01.bat` | 从零启动固定配置训练，拒绝覆盖已有目录 |
| `docs/LEARNING_MODE_0.1.md` | 学习课程、语料适配及其限制 |
| `docs/EXPERIMENT_EVIDENCE_0.1.md` | 小问答实验及复现记录 |
| `release-0.1-retained-sources.json` | 保留源文件清单 |
| `build/` | 当前数据、权重、运行日志及本地可执行文件 |
| `VERSION` | 实现版本0.1.0 |

旧源码、旧实验与历史模型移出工作树，归档位置及恢复清单见 `docs/CLEANUP_0.1.json`。归档不是永久删除。

## 构建

已验证环境：Windows x64，Visual Studio Enterprise 18 / MSVC14.44，CUDA12.6；CUDA目标 `sm_89`，CPU AVX2。需要相应编译器、CUDA运行时；构建脚本中的工具链路径按本机配置填写。

```bat
cd /d D:\TaoVm
build_project.bat
```

生成 `build/yaoyao_train_v01.exe` 和 `build/yaoyao_cpu_v01.exe`。新名称避免覆盖当前运行中的训练二进制。运行中的历史命名 `train_noffn_fresh.exe` / `cpu_noffn.exe` 同样使用0.1架构。编译不会自动替换已运行进程。

语料重新导出使用 `src/export_repair_bpe_train.cpp`，额外需要 Apache Arrow/Parquet C++ 环境；基础构建不自动下载数据或重新导出。

## 数据与训练

现有训练输入：`build/repair_export_20260909_v2/train.bin`，TLP2 格式，19,073 个完整对话，5,953,644 tokens，其中3,887,672个监督目标。SHA256：

```text
6611410defa4fb41ca22f73ae4ee5b8ff51df0803a6437826c0bf94757e2899f
```

冻结 tokenizer：`build/formal_tokenizer.bbp`；特殊ID：BOS256、USER257、ASSISTANT258、TURN_END259、EOS260。仅 assistant 内容及其结束标记参与相应监督。训练入口校验语料manifest、接受receipt、tokenizer及固定验证／测试数据身份。结构验收不等于答案内容质量认证。

| 训练配置 | 值 |
|---|---|
| 初始化 | seed20260911，权重全新，Adam与状态为零 |
| 目标 | 2000 次优化器更新，不是2000次全语料遍历 |
| GPU模式 | 4个槽位 × 每更新8次图重放 × 256步TBPTT |
| 优化器 | AdamW，beta1=.9、beta2=.999，矩阵decay=.01 |
| 梯度 | 监督目标数归一化，范数裁剪至1，三值投影/STE |
| 学习率 | 前100步预热至.00025；1500至2000余弦降至.00005 |
| 课程 | 前100步完整短对话（≤256 tokens），随后全部合格对话 |
| 固定验证 | 每次更新后，在GPU上计算固定NLL |
| 保存 | step0，之后每100步；SCP、游标状态和CPU用DSB |

固定验证输入 `build/bpe_pilot_validation.bin`：23篇，4906个监督目标；测试文件 `build/bpe_pilot_test.bin`。课程切换在step100保存后进行，重置数据游标及会话状态，保留新训练已经学习的权重和Adam。

小样本中验证的模式是“先干净后混合变化”；真实语料没有盲目换序、插噪声或改写答案。本轮短对话到完整语料的适配尚未证明复制小实验收益。

```bat
rem 仅在没有当前训练目录/日志时启动
scripts\train_v01.bat
```

当前固定输出：

- 日志：`build/noffn_fresh.log`
- checkpoint：`build/noffn_fresh/step_100/` 等
- CPU权重：对应目录的 `final.dsb`
- 最终权重：完成2000步后 `build/noffn_fresh/final.dsb`

训练器自行打开固定日志并刷新。存在现有输出时禁止自动覆盖或重新启动。当前可执行入口为fresh-only；SCP保留优化器与游标信息，但官方脚本尚未提供通用恢复命令，不能仅凭有SCP宣称一键续训可用。

要请求在下一优化器边界保存并停止，创建 `build/STOP_NOFFN`。`build/STOP_TRAINING` 必须保留：本训练路径沿用repair-mode许可约束，它不是此路径的停止开关。不要通过删除它来启动训练。

## CPU对话

```bat
build\yaoyao_cpu_v01.exe build\noffn_fresh\step_100\final.dsb
```

逐行输入问题；`/reset`清空会话状态，`/quit`退出。权重只加载一次，最长回复64 tokens；遇到TURN_END/EOS结束。这里进行CPU推理，不进行CPU神经训练。

step100速度测试：模型加载约105ms；忽略结束标记连续贪心计算，5×512步汇总约589 token/s；正常10-token提示平均14.31ms但返回空回复。因此吞吐是计算指标，不是有效文本生成能力。测试源码 `src/benchmark_noffn_decode.cpp`，记录 `build/noffn_step100_decode_speed.log`。

## 验证证据与限制

从零小模型实验：无FFN、2层d64，两个新种子，训练／开发／封存测试组合隔离。干净训练与课程训练在封存测试上的描述性汇总准确率54.46%→88.84%；叠加变化仍只有65.63%。相同测试题跨种子重复，不能视为更多独立题目；不是自然语言能力证明。

训练后小模型CPU/CUDA连续24步贪心选择一致，最大logit误差≤2.87e-6。生产0.1新DSB可由新CPU解码器加载，旧解码器拒绝。step100生产模型九次问答为空，当前尚不能称为可用助手。

当前运行日志里遗留的 `backend=arch2`、初始 `adam_reset=0` 和初始LR常量是旧标签；实际架构以DSB/SCP算子身份、schema及正式构建宏为准。不会为了改标签中断正在训练的进程。

## 0.1版本边界

本次发布整理本地源码、构建入口和文档，不发布远端仓库、不自动打Git标签、不承诺训练完成或对话能力。训练中的权重、数据、固定日志及STOP文件保留；旧内容可从清理清单恢复。
