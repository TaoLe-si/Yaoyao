# 夭夭 (Yaoyao) v0.6 架构文档与数学原理

## 文档信息

| 项 | 值 |
|---|-----|
| 版本 | v0.6 (clean, no vocab layer) |
| 实现 | 单一 CPU 实现: yaoyao_v06_cpu.cpp |
| 状态 | 已回退 v0.6 baseline, 移除 GPU 训练路径, 移除词汇层 |
| 编译 | clang++ -O2 -std=c++17 -mavx2 -mfma -fopenmp |
| 数据 | TinyStories 训练集 (478M tokens) |

---

## 一、整体架构图

    输入字符 id (BATCH x SEQ, 共 2048 个 token)
                          |
                          v
             [1] Embedding  (V=142 -> D=256)
                          |
                          v
    +-----------------------------------------------+
    |  Layer l = 0..5  (N_LAYERS = 6)               |
    |                                               |
    |   [2] Q3 卷积  (k=3, in-place)               |
    |        |                                      |
    |        v                                      |
    |   [3] Dynamic Alpha  (z -> sigmoid(z/T))      |
    |        |                                      |
    |        v                                      |
    |   [4] Channels  (h 衰减 + s 累加)             |
    |        |                                      |
    |        v                                      |
    |   [5] RMSNorm  (per-row)                      |
    |                                               |
    |  Residual: x_l+1 = hg + sg                    |
    +-----------------------------------------------+
                          |
                          v
    +-----------------------------------------------+
    |  [6] Q4 字符头 (跨层共享)                    |
    |      logits[v] = W_h*hg + W_s*sg + W_bi[prev,v] |
    |      (无 bias)                                |
    +-----------------------------------------------+
                          |
                          v
                [7] LogSoftmax + NLL (per token)
                          |
                          v
                       Loss

---

## 二、各模块精确数学

### [1] Embedding

    x_t[d] = Emb[input[t]][d],  d in [0, D), t in [0, SEQ)

- V = 142 (TinyStories 字符表: a-z, A-Z, 0-9, 标点, 空格 + pad/unk)
- D = 256 (隐藏维度)
- Emb in R^(VxD)
- 初始化: N(0, 0.5^2)
- Adam, clip +/- 4

---

### [2] Q3 卷积 (k=3)

    y_t[d] = clamp( w_0[d]*x_{t-2}[d] + w_1[d]*x_{t-1}[d] + w_2[d]*x_t[d], -4, +4 )

- 边界: t<2 时 w_0 项忽略; t<1 时 w_1 项忽略
- Dropout (训练时): 对每个 d 以概率 0.1 置 0, 激活的 y 乘 1/(1-0.1)
- w_0, w_1, w_2 in R^D (每层一组)
- 初始化: N(0, (0.1*0.3)^2)
- Adam, clip +/- 2

---

### [3] Dynamic Alpha (输入依赖遗忘率)

    z_t[d] = b_alpha[d] + sum_{k=0..D-1} W_alpha[d,k] * x_t[k]

    alpha_t[d] = sigma(z_t[d] / T) = 1 / (1 + exp(-z_t[d] / T)),  T = 2.0

- W_alpha in R^(DxD) (每层)
- 正交初始化: 每行从 {+/-1}^D 随机选取并归一化 (DeepSeek 推荐)
- b_alpha[d] = -1.7 + N(0, 0.1^2)  ->  sigma(-1.7/2) ~= 0.30
- Adam, clip W_alpha +/- 4, clip b_alpha +/- 8

T=2.0 的意义: 温度放大 z 让 sigmoid 工作在线性区. 直觉: T=1 时 alpha 二值化严重 (0/1), T=2 让 alpha 在 0.2~0.8 之间平滑过渡, 适合学习"逐 token 渐变"的衰减率.

---

### [4] Channels (双通道)

    h_0[d] = 0,    s_0[d] = 0
    h_{t+1}[d] = alpha_t[d]*h_t[d] + (1 - alpha_t[d])*y_t[d]
    s_{t+1}[d] = s_t[d] + y_t[d]

- h 维度 [BATCH x (SEQ+1) x D] (含 t=0 初始 0)
- s 同上
- 数值范围: y 被 clamp 到 +/-4, 因此 |s_{SEQ}| <= SEQ*4 = 256
- h 是 EMA, 范围 <= 4

---

### [5] RMSNorm

    hbar_t[d] = h_{t+1}[d] / sqrt( (1/D) * sum_k h_{t+1}[k]^2 + eps ),  eps = 1e-5
    sbar_t[d] = s_{t+1}[d] / sqrt( (1/D) * sum_k s_{t+1}[k]^2 + eps )

- per-row normalize (每 token 独立归一化)
- 输出进入 Q4

---

### [6] Q4 字符头

    logits[v] = sum_d W_h[v,d]*hbar[d] + sum_d W_s[v,d]*sbar[d] + W_bi[prev_token, v]

- W_h, W_s in R^(VxD) (跨层共享, 即最后一层输出直接投影)
- W_bi in R^(VxV), bigram 矩阵
- prev_token = input[t-1] (t>0) 或 pad_id (t=0)
- Bias 禁用: logits 不加 bias
- W_h, W_s 初始化 N(0, 0.1^2), Adam 后 clip +/- 1
- W_bi 初始化 0, Adam 后 clip +/- 8

---

### [7] LogSoftmax + NLL

    m = max_v logits[v]
    log_Z = m + log sum_v exp(logits[v] - m)
    P(v) = exp(logits[v] - log_Z)
    L_n = -logits[target_n] + log_Z
    L = (1/N) sum_n L_n

梯度: dL_n/dlogits[v] = P(v) - 1[v == target_n]

---

## 三、训练目标与反向传播

训练目标: 最小化 L (字符级 NLL).

优化器: Adam (beta1=0.9, beta2=0.999, eps=1e-8)

两套学习率:
- LR_main = 0.005 (Embedding, W_h, W_s, W_bi, W_q3, bias)
- LR_alpha = 0.0005 (W_alpha, b_alpha, 主 LR 的 1/10)

学习率调度:
- Warmup (1 epoch): 线性从 0 到 LR
- Cosine decay (剩余 epochs): LR_t = LR_min + 0.5*(LR_max - LR_min)*(1 + cos(pi*progress))

梯度裁剪: 全局 L2 范数 <= 1.0 (本实现用 per-tensor clip +/- 1)

---

## 四、反向传播 (Full BPTT, 6 层完整链路)

### 4.1 字符头反向

    dL/dlogits[v] = P(v) - 1[v == target]

    dL/dW_h[v,d] = sum_n dL/dlogits[n,v] * hbar[n,d]
    dL/dW_s[v,d] = sum_n dL/dlogits[n,v] * sbar[n,d]
    dL/dW_bi[u,v] = sum_{n: prev(n)=u} dL/dlogits[n,v]

### 4.2 Q4 输入梯度

    dL/dhbar[d] = sum_v dL/dlogits[v] * W_h[v,d]
    dL/dsbar[d] = sum_v dL/dlogits[v] * W_s[v,d]

### 4.3 RMSNorm 反向

    r = sqrt( (1/D) sum_k x_k^2 + eps )
    r_dot = sum_k (dL/dy_k)*x_k / (r^3 * D)
    dL/dx_d = (1/r) * dL/dy_d - x_d * r_dot

### 4.4 Channels 反向 (时间反传)

已知 dL/dh_{t+1}, dL/ds_{t+1}:

    dL/dy_t     = (1 - alpha_t) * dL/dh_{t+1} + dL/ds_{t+1}
    dL/dalpha_t = (h_t - y_t) * dL/dh_{t+1}
    dL/dh_t     = alpha_t * dL/dh_{t+1}
    dL/ds_t     = dL/ds_{t+1}

边界: dL/dh_{SEQ} = dL/ds_{SEQ} = 0

### 4.5 Alpha 反向

由 alpha = sigma(z/T):

    dL/dz = dL/dalpha * alpha*(1-alpha) / T
    dL/dW_alpha[d,k] = sum_t dL/dz_t[d] * x_t[k]
    dL/db_alpha[d]    = sum_t dL/dz_t[d]
    dL/dx_t[k]       += sum_d dL/dz_t[d] * W_alpha[d,k]

### 4.6 Q3 反向

    dL/dx_{t-2}[d] += dL/dy_t[d] * w_0[d]    (t>=2)
    dL/dx_{t-1}[d] += dL/dy_t[d] * w_1[d]    (t>=1)
    dL/dx_t[d]     += dL/dy_t[d] * w_2[d]
    dL/dw_0[d] = sum_{t>=2} dL/dy_t[d] * x_{t-2}[d]

(其他两个权重同理)

### 4.7 Embedding 反向

    dL/dEmb[u,d] = sum_{n: input[n]=u} dL/dx_n[d]

---

## 五、模型规模与超参数

| 参数 | 形状 | 大小 | 初始化 | clip |
|------|------|------|--------|------|
| Embedding | 142 x 256 | 36,352 | N(0, 0.5^2) | +/- 4 |
| W_h | 142 x 256 | 36,352 | N(0, 0.1^2) | +/- 1 |
| W_s | 142 x 256 | 36,352 | N(0, 0.1^2) | +/- 1 |
| W_bi | 142 x 142 | 20,164 | 0 | +/- 8 |
| W_q3,0/1/2 | 6 x 256 each | 1,536 each | N(0, 0.03^2) | +/- 2 |
| W_alpha | 6 x 256 x 256 | 393,216 each | 正交 +/- 1/sqrt(D) | +/- 4 |
| b_alpha | 6 x 256 | 1,536 each | -1.7 + N(0, 0.1^2) | +/- 8 |

总参数: ~ 1.0 M float32 ~= 4 MB

Adam moments: x2 = ~ 8 MB (运行时)

---

## 六、超参数

| 项 | 值 |
|----|-----|
| V | 142 |
| D | 256 |
| SEQ | 64 |
| BATCH | 32 (BL = 2048 tokens) |
| N_LAYERS | 6 |
| N_WIN (per epoch) | 30,000 |
| EPOCHS | 12 |
| LR_main | 0.005 |
| LR_alpha | 0.0005 |
| LR_min | 0.0001 |
| T_alpha | 2.0 |
| Dropout | 0.1 |
| Max grad norm | 1.0 |
| beta1 | 0.9 |
| beta2 | 0.999 |
| Adam eps | 1e-8 |

---

## 七、关键设计决策（为什么这样设计）

### 7.1 双通道 h + s

- h 通道 (EMA): 短时记忆, 受 alpha 控制, 用于"近期 token 的衰减平均"
- s 通道 (累积): 长时累加, 永不衰减, 用于"整句的统计信息"

两者通过 RMSNorm 平衡量级后, 共同进入 Q4 head.

### 7.2 Dynamic Alpha (vs. Static 衰减)

- Static (e.g., GRU): 衰减率固定, 无法适应不同语境
- Dynamic (DeepSeek/RWKV 风格): alpha_t 由输入决定, 不同 token 可以有不同"记忆时长"
- 温度 T=2: 防止 alpha 二值化 (0/1), 保留梯度流

### 7.3 Q3 卷积 (k=3) 替代 token-level 注意力

- Q3 是逐通道 1D 卷积, 参数量 = 3D (vs. attention 的 D^2 或 4D^2)
- 上下文长度 = 3 (即 x_{t-2}, x_{t-1}, x_t)
- 配合 h 通道的全局衰减, 等效于"局部 + 全局" 的混合

### 7.4 6 层堆叠 (N_LAYERS=6)

- 每层增加一次"局部-全局混合"
- 6 层足够学习 TinyStories 的语法结构
- 总参数 ~ 1M, 训练 1 epoch ~ 30 min (CPU, AVX2)

### 7.5 Bigram W_bi (无 vocab layer 时唯一的位置先验)

- 没有位置编码 (no positional embedding)
- 没有 RNN/LSTM
- Bigram 是最简单的"位置先验": logits[v] 受 input[t-1] 直接影响
- 等价于一个"per-token 的 language model head"
- 单独 W_bi 矩阵 20K 参数, 占用小但提供强先验

### 7.6 Bias 禁用

- 实验发现 bias 项容易让 logits 漂移到无界, 需要更多 clip
- 移除后训练更稳定, 表达力由 weight matrix 承担

### 7.7 Adam 双重 LR (main vs. alpha)

- Alpha 参数极度敏感: alpha 直接决定 h/s 的衰减率
- 同样大小的梯度在 alpha 上比在 W_h 上产生大得多的"行为变化"
- 经验: LR_alpha = LR_main / 10 是最佳比例

---

## 八、训练 / 推理流程

### 训练 (CPU)
1. 加载 vocab (vocab.txt) 和训练文本 (tinystories_train.txt)
2. UTF-8 -> token ids (142-token char vocab)
3. 按 30K 个窗口 (SEQ=64) 切分
4. 每个 epoch:
   - 每个 batch: forward (6 layers) -> NLL loss -> backward (6 layers BPTT) -> Adam
   - 每 100 batch 打印 loss
5. 训练结束保存 (后续工作)

### 推理 (CPU, 当前 main 中 gen_one)
- 单 token 顺序生成
- 每步: forward 6 层 -> argmax over logits
- 局限: 6 层递归 forward, 每步 O(D^2) = O(65K) 操作
- 优化: 缓存 h/s (实际生成时只算新 token), 暂未实现

---

## 九、文件清单

| 文件 | 描述 |
|------|------|
| yaoyao_v06_cpu.cpp | 干净的 v0.6 CPU 实现 (训练 + 推理) |
| yaoyao_v06_cpu.exe | 编译产物 (Windows + clang++) |
| yaoyao_v06_cpu_train.log | 训练日志 |
| vocab.txt | 142-token 字符表 |
| tinystories_train.txt | 478M char 训练语料 |

---

## 十、已删除文件 (回退)

| 文件 | 原因 |
|------|------|
| yaoyao_gpu.cu | GPU 训练路径 (用户删除) |
| yaoyao_gpu_v07.cu | GPU v0.7 训练 |
| yaoyao_gpu_train_v07.cu | GPU v0.7 训练副本 |
| yaoyao_v11.exe, yaoyao_v10b.exe | GPU 训练 binary |
| yaoyao_v0_7.cpp, yaoyao_v0_7_smoke.cpp | v0.7 (含 vocab layer) 备份 |
| yaoyao_v07_smoke.exe | v0.7 smoke binary |
| yaoyao_math_test.cpp | 数学测试程序 |

架构变更总结: 完全回退到 v0.6 baseline, 删除词汇层和 GPU 路径. 当前架构为 6 层 Q3+DynamicAlpha+Channels+Q4+Bigram, 字符级 NLL, 单一 CPU 实现.

---

## 附: 数学符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| B | batch size | 32 |
| S | sequence length | 64 |
| D | hidden dim | 256 |
| V | vocab size | 142 |
| L | num layers | 6 |
| N | total tokens in batch = B*S | 2048 |
| T | alpha temperature | 2.0 |
| eps | RMSNorm eps | 1e-5 |
| x_t | layer input at token t | [D] |
| y_t | Q3 output (clamped) | [D] |
| alpha_t | per-dim forget rate | [D] |
| h_t, s_t | channel states | [D] |
| hbar_t, sbar_t | RMSNorm outputs | [D] |
| logits[v] | Q4 output | [V] |
| W_h, W_s | Q4 weights | [VxD] |
| W_bi | bigram weight | [VxV] |
| W_alpha, b_alpha | alpha dynamics | [DxD], [D] |
| W_q3,0/1/2 | Q3 conv weights | [D] each |
| Emb | embedding | [VxD] |

---

## 附: 编译运行命令

    clang++ -O2 -std=c++17 -mavx2 -mfma -fopenmp \
        -o yaoyao_v06_cpu.exe yaoyao_v06_cpu.cpp

    ./yaoyao_v06_cpu.exe

输入: vocab.txt, tinystories_train.txt
输出: yaoyao_v06_cpu_train.log (含 loss, 生成样本)