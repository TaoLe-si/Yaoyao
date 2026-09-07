# CPU 原生大模型架构（自研 · 架构设计问卷）

> 状态：v0.2（已纳入 Q2-A + Sum 通道 + 多项式头，含实测数据）
> 目标读者：作者本人 + 后续协作的工程/研究同事
> 维护原则：每条设计都要回答"它比 Transformer 省了哪类搬运"

---

## 0. 立铁律（写在最前面）

CPU 的核心痛点是 **内存带宽（搬运数据）**，而不是峰值算力。

因此整套架构的设计目标是：

> **每一步都尽可能少搬动大矩阵，尽量重用缓存里的数据。**

后面所有取舍，都以此为裁判。

---

## 1. 架构设计问卷：四个核心问题

| 编号 | 问题 | Transformer 的做法 | 自研架构的方向 |
|------|------|--------------------|------------------|
| Q1 | 怎样给输入的数据做"身份" | 查大词表 + 位置编码 | 压缩与固化（**已锁**：哈希桶池） |
| Q2 | 怎样处理"输入"与"记忆"的概率 | Q·K 点积 + KV Cache | 固定大小的状态记忆（**已锁**：Q2-A + Sum） |
| Q3 | 怎样处理"数据与数据之间"的概率 | 多头注意力 O(n²) | 局部卷积 + 衰减递推 |
| Q4 | 怎么输出 | 大矩阵 × 隐藏状态 | 缩小战场（**已锁**：多项式头） |

下面逐题展开，**已锁定的设计会标注 ✅ 实验数据**。

---

## 2. Q1 · Input Encoding & Embedding ✅ 已锁定

### 2.1 设计 · 哈希桶池 (Hash Bucket Pool)

```
输入: token_id ∈ [0, V)
     ↓
哈希函数 H_1, H_2, ..., H_K    K 个独立哈希
     ↓
x[d] = bucket[H_k(token_id) % N_BUCKETS][d]   从桶池取 K 行
```

**关键参数**：
- 桶池大小 `[H, D]`（H = 桶数，D = 模型维度）
- 每个 token 实际由 **K 个桶位置** 拼出
- 桶内值为 **trit {-1, 0, +1}** 或 **小整数 {-K_VAL, ..., +K_VAL}**

**优势**：
| | 传统词嵌入 | 哈希桶池 |
|--|--|--|
| 表大小 | `[V × D]` | `[H × D]`，H << V |
| 内存 (D=4096, V=50K) | 200 MB | **16 MB**（trit 打包） |
| 查表 | 1 次大表读 | K 次小桶读（可并行） |
| Trit 打包后 | 200 MB | **4 MB** |

**Q1 已锁定为方案 1**（K=3-4 个桶拼出 token）。

### 2.2 数学公式

```
令 hash_k(id) = (id · salt_k) mod H,    salt_k 是预生成随机种子

x_t[d] = sum_{k=1..K} B[hash_k(token_id), d]
```

- `B[h, d]` 是桶池参数，训练时学习
- 输出 `x_t ∈ {-K, ..., +K}^D`（小整数向量）

---

## 3. Q2 · "输入"与"记忆"的概率  ✅ 已锁定（Q2-A + Sum 通道）

### 3.1 核心洞察

**单一通道不够**——必须有多个**互补**的状态通道：

| 通道 | 类型 | 作用 |
|------|------|------|
| `h` (Q2-A 主通道) | EMA 有界 `[H_VAL]` | **近期/位置敏感**任务 |
| `s` (Sum 通道) | 累加 有界 `[S_MAX]` | **全局/顺序无关**任务 |

### 3.2 Q2-A 主通道（EMA）

**更新公式**：
```
h_new[d] = clamp(round(α[d]·h_old[d] + (1-α[d])·x_t[d]), ±H_VAL)
```

- `h ∈ [-H_VAL, +H_VAL]^D`，**int8** 存储（H_VAL=4 时）
- `α ∈ [0, 1]^D`，**per-dim 学习的衰减率**
- 每步 O(D) ops，**纯整数运算**

**擅长**：
- "最后一个 token"（α 小）
- "最近 token 趋势"（α 中）
- 部分顺序敏感任务

**不擅长**：
- 全局聚合（顺序无关）：EMA 权重对位置倾斜，**做不到真正的 sum/avg**

### 3.3 Sum 通道（关键补全）✅ 实验验证

**更新公式**：
```
s_new[d] = clamp(s_old[d] + x_t[d], ±S_MAX)
```

- `s ∈ [-S_MAX, +S_MAX]^D`，**int16** 存储
- **无衰减，无乘法**，纯加法
- S_MAX 由序列长度决定（N × |x_max|）

**为什么必须**：
- `h` 通道做 EMA，权重对位置倾斜（α=0.9 时最近 token 权重是第 1 个的 5×）
- α=0.99 时权重接近均匀但信号极弱（h_max ≈ 0.5）
- **`sum` 是唯一天生顺序无关的算子**

**实验验证**（多数任务）：
```
D=4096, 16-token 序列, 多数 A vs 多数 B
Q2-A only (α=0.99):   66% 卡住
Q2-A + Sum:           100% @ epoch 1
```

### 3.4 两个通道的状态大小

```
h: D bytes (int8)         4 KB @ D=4096
s: 2D bytes (int16)       8 KB @ D=4096
x_t: D bytes (int8)       4 KB @ D=4096
α: 4D bytes (float)      16 KB @ D=4096

总计：32 KB 工作集 → 完全在 L1 缓存
```

---

## 4. Q3 · "数据与数据之间"的概率

### 4.1 自研设计 · 三层互补

| 层 | 机制 | 作用 | 状态 |
|----|------|------|------|
| 局部 | 短卷积 k=3 或 5 | 邻域语法 | 待实现 |
| 近期 | Q2-A 的 `h` 通道 | 末段趋势 | ✅ 验证 |
| 全局 | Q2-A 的 `s` 通道 | 累加聚合 | ✅ 验证 |

### 4.2 Sum 通道的实证（**关键实验**）

**Task A**：`count_a == 8`（s=0） vs `count_a ∈ {4, 12}`（s=±24）

| 头类型 | 准确率 | 备注 |
|--------|--------|------|
| 线性头 | 87-90% | 卡住：线性头不能区分 s=0 和 s=+24 |
| **多项式头 (含 s²)** | **100%** | s²=0 vs s²=576 干净分开 |

**Task B**：XOR(majority, last_token)

| 头类型 | 准确率 | 备注 |
|--------|--------|------|
| 线性头 | 46-53% | 卡住：XOR 非线性 |
| **多项式头 (含 h·s)** | **100%** | 同号 → label 0，异号 → label 1 |

---

## 5. Q4 · Output Head ✅ 已锁定（多项式头）

### 5.1 设计 · 多项式特征头

**输入**：`h ∈ ℤ^D`, `s ∈ ℤ^D`（两个通道的最终状态）

**特征**：
```
[h, s, s², h·s]      共 4 个 D 维特征
```

**输出**：
```
logit = w_h·h + w_s·s + w_s2·s² + w_hs·(h·s) + b
```

### 5.2 为什么多项式而不是 MLP？

| | 多项式头 | MLP (hidden=16) |
|--|--|--|
| 额外参数 | 2D | 32D |
| 参数量 | 8K | 131K |
| 表达力 | 限定二次形式 | 任意 |
| 适用于 | 多数聚合 + 简单非线性 | 复杂组合 |

**多项式对 sum 通道的天然适配**：s² 自动检测"sum 接近 0"。

### 5.3 全任务能力表

| 任务 | 通道 | 特征 | α (学到) | 准确率 |
|------|------|------|---------|--------|
| 多数 (majority) | s | s | 0.99 | 100% |
| 精确数 count=8 | s | **s²** | 0.99 | 100% |
| 最后 token | h | h | 0.05 | 100% |
| XOR(maj, last) | **s + h** | **h·s** | **0.05** | 100% |

**关键**：模型自己学到 α——同一架构既能"长期聚合"又能"短期位置"。

---

## 6. 终极蓝图 · v0.2 单层结构

```
输入 token_id
     ↓ Q1 哈希桶池 (16 MB 参数)
x_t ∈ {-K, ..., +K}^D           4 KB
     ↓
┌──────────────────────────────────────────┐
│  Q2-A 状态递推                              │
│                                           │
│  h_new[d] = round(α[d]·h[d] + (1-α[d])·x_t[d])  │
│  s_new[d] = s[d] + x_t[d]                 │
│                                           │
│  h: int8  [-4, +4]^D     4 KB              │
│  s: int16 [-64, +64]^D   8 KB              │
│  α: float [0, 1]^D       16 KB             │
└──────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────────┐
│  多项式头                                  │
│  logit = w_h·h + w_s·s + w_s2·s² + w_hs·(h·s) + b │
└──────────────────────────────────────────┘
```

**单层参数 + 状态：**
- 桶池 Q1：16 MB（共享于所有层）
- α (per-layer)：16 KB
- head (per-layer)：16 KB（4 × D × 1 byte trit）
- 状态 h + s：12 KB（per-token）
- **总计：~16 MB 参数 + 12 KB per-token 状态**

---

## 7. 与现有工作的对齐

| 工作 | 对齐到自研蓝图 |
|------|----------------|
| **Mamba**（SSM） | Q2 状态递推（h 通道），但**没 Sum 通道**——Mamba 没法严格顺序无关聚合 |
| **RWKV**（WKV） | Q2 标量门控（α），但也没 Sum 通道 |
| **LFM2**（混合架构） | Q3 短卷积 + 主干递推——我们用 Sum 通道替代其长程依赖 |
| **低比特量化**（GPTQ / AWQ） | Q1 词表量化、Q4 head 量化 |
| **GLU / SwiGLU**（激活函数） | **不需要**——h 通道靠 α 门控自然实现非线性 |

**自研蓝图的独特点**：
- **Sum 通道**——比 Mamba/RWKV 多一个**严格顺序无关**的聚合器
- **多项式头**——比线性头多 2 个二次特征，参数只多 2D
- **per-dim 自适应 α**——比 RWKV 的标量衰减更灵活

---

## 8. 实测性能（D=4096, 单核, -O2 -march=native）

### 8.1 单步延迟

| 配置 | µs/step | ns/dim |
|------|---------|--------|
| Q2-A only | 16.49 | 4.02 |
| Q2-A + Sum | 21.34 | 5.21 |
| **Q2-A + Sum + Poly** | **20.60** | **5.03** |

### 8.2 吞吐量与扩展

```
D=4096, SEQ=1K:   22 ms/seq  →  46K tokens/s
D=4096, SEQ=4K:   85 ms/seq  →  47K tokens/s   ← 长度无关
D=8192, SEQ=1K:   54 ms/seq  →  19K tokens/s
```

### 8.3 内存

```
D=4096:  96 KB 工作集   (L2 友好)
D=8192: 192 KB 工作集   (L2 友好)
```

### 8.4 vs Transformer 单层

| | Transformer (D=4096) | Q2-A + Sum + Poly |
|--|--|--|
| Per-layer per-token | ~9 ms | **20 µs** |
| 倍数 | 1× | **~450× 更快** |

### 8.5 27B 类模型估算（32 层 × D=4096）

- 单 token 单线程：**0.64 ms** → ~1,500 tokens/s
- 32 路层并行：**0.02 ms** → ~50,000 tokens/s
- 每 token 状态：**12 KB**（vs Transformer 32 MB+ KV cache）
- 内存效率：**~2,500×** 优于 Transformer

---

## 9. 模型能力矩阵（已验证）

| 任务类型 | 例子 | 准确率 | 通道 |
|---------|------|--------|------|
| 近期位置 | 预测最后 token | 100% | h |
| 全局聚合 | 多数 (majority) | 100% | s |
| 精确数量 | count_a = 8 | 100% | s + s² |
| 跨通道组合 | XOR(maj, last) | 100% | h + s + h·s |

---

## 10. 待办与下一步

### 已完成 ✅
- [x] Q1 哈希桶池设计 + 实现
- [x] Q2-A 公式 + 测试
- [x] **Sum 通道**——突破聚合瓶颈
- [x] **多项式头**——解决非线性分类
- [x] 性能基准（per-step, scaling）
- [x] vs Transformer 对比
- [x] 架构文档 v0.2 整理

### 待完成
- [ ] **SIMD (AVX2) 优化**——目标 4× 加速比
- [ ] **多层级联**——32 层 27B-class 端到端测试
- [ ] **更复杂聚合任务**——OR-of-AND, parity, max-pool
- [ ] Q3 局部卷积与 Q2-A 串联/并联
- [ ] 实际长文本训练（Q1 + Q2-A 串起来）

---

## 11. 变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 初稿 | 建立四问四答框架与终极蓝图，尚未填公式 |
| v0.2 | 实测版 | + Q1 哈希桶池锁定，+ Q2-A + Sum 双通道，+ 多项式头，+ 实测数据 ### Q4: Output Decoding (v0.4 - AVX2 Hash Bucket)

**Problem**: Full `[V, D]` W matrix is 200 MB for V=50K, D=4096 → memory-bound (~10 GB/s, 100ms).

**Solution**: Hash Bucket Q4 — shared 4MB pool, fits L2/L3 cache.

**For binary classification (V=2)**: Polynomial feature head:
```
logit = w_h·h + w_s·s + w_s2·s² + w_hs·(h*s) + b
```

**For multi-class (V ≥ 50)**: Hash Bucket Q4 with AVX2:
```
Pool: [N_BUCKETS=1024, D=4096] = 4 MB (L3 cache)
For each token v: hash(v, k) = (v * K_HASH + k) % N_BUCKETS

1. Compute bucket_scores[b] = Pool[b] · state  (N_BUCKETS * D ops)
2. logit_v = bias[v] + sum_{k=0..K_HASH-1} bucket_scores[hash(v, k)]
3. Top-K = argmax of logit_v via AVX2 batched max + linear scan
4. (Optional) Verify top-K candidates with full V projection
```

**Performance** (V=50K, D=4096, AVX2):
| Method | Time | Speedup |
|--------|------|---------|
| Naive linear | 182 ms | 1× |
| AVX2 sort | 35 ms | 5× |
| Top-K Sparse (still reads V×D) | 27 ms | 7× |
| **Hash Bucket Q4** | **553 µs** | **329×** ✅ |

### Channel-Task Mapping

Each Q4 head uses ONE primary channel:
| Task | Channel | Reason |
|------|---------|--------|
| Predict majority | `s` (sum) | Global count |
| Predict next token | `h` (EMA) | Local context |
| Detect rare token | `m` (max) | Any-occurrence |
| Detect alternation | `p` (parity) | Sign pattern |

Separate heads (one per task) share channels but specialize.

### 夭夭 Yaoyao v0.1 - Real Text (TinyStories)

Multi-layer Q1+Q3+Q2-A+Sum+Q4, Adam, char-level.

- Loss: 4.96 -> 2.55 (10 epochs)
- Speed: 15,808 tokens/s CPU
- Architecture: 4-layer stack, D=64, V=142
- File: D:\TaoVm\yaoyao_v0_1.cpp

### Basic LM Validation (v0.5 update)

**Proof of concept**: Input question → output answer with full accuracy.

Test: 10 digit-word pairs trained end-to-end.
- Char-level vocab (39 chars)
- D=32, trainable embeddings + sum channel + linear Q4
- **Optimizer: Adam (lr=0.01)**
- **Result: 10/10 (100%) accuracy** after 600 epochs
- Loss: 4.07 (random) → 0.0002

```
Input: "0:" -> "zero"     Input: "5:" -> "five"
Input: "1:" -> "one"      Input: "6:" -> "six"
Input: "2:" -> "two"      Input: "7:" -> "seven"
Input: "3:" -> "three"    Input: "8:" -> "eight"
Input: "4:" -> "four"     Input: "9:" -> "nine"
```

### Q4 Benchmarks (Final, v0.4)

- Binary (polynomial): 2.00 µs (D=4096)
- V=8 multi-task (2 heads): 0.34 µs/total
- Hash Bucket Q4 V=50K: **553 µs**
- Full pipeline 32 layers + Q4: **698 µs/token = 1,431 tokens/s**
- vs user Qwen 35B-A3B MOE (20 t/s): **71.6× faster**

### Why Hash Bucket Wins

Naive Linear Q4 reads 200MB W matrix per token → memory bound.
Hash Bucket reads only 4MB shared pool → fits in L3.
Bucket aggregation gives `O(V * K_HASH)` extra ops but cached.
329× speedup from avoiding memory bandwidth wall.


## 0. 立铁律（写在最前面）

CPU 的核心痛点是 **内存带宽（搬运数据）**，而不是峰值算力。

因此整套架构的设计目标是：

> **每一步都尽可能少搬动大矩阵，尽量重用缓存里的数据。**

后面所有取舍，都以此为裁判。

---

## 1. 架构设计问卷：四个核心问题

| 编号 | 问题 | Transformer 的做法 | 自研架构的方向 |
|------|------|--------------------|------------------|
| Q1 | 怎样给输入的数据做"身份" | 查大词表 + 位置编码 | 压缩与固化（**已锁**：哈希桶池） |
| Q2 | 怎样处理"输入"与"记忆"的概率 | Q·K 点积 + KV Cache | 固定大小的状态记忆（**已锁**：Q2-A + Sum） |
| Q3 | 怎样处理"数据与数据之间"的概率 | 多头注意力 O(n²) | 局部卷积 + 衰减递推 |
| Q4 | 怎么输出 | 大矩阵 × 隐藏状态 | 缩小战场（**已锁**：多项式头） |

下面逐题展开，**已锁定的设计会标注 ✅ 实验数据**。

---

## 2. Q1 · Input Encoding & Embedding ✅ 已锁定

### 2.1 设计 · 哈希桶池 (Hash Bucket Pool)

```
输入: token_id ∈ [0, V)
     ↓
哈希函数 H_1, H_2, ..., H_K    K 个独立哈希
     ↓
x[d] = bucket[H_k(token_id) % N_BUCKETS][d]   从桶池取 K 行
```

**关键参数**：
- 桶池大小 `[H, D]`（H = 桶数，D = 模型维度）
- 每个 token 实际由 **K 个桶位置** 拼出
- 桶内值为 **trit {-1, 0, +1}** 或 **小整数 {-K_VAL, ..., +K_VAL}**

**优势**：
| | 传统词嵌入 | 哈希桶池 |
|--|--|--|
| 表大小 | `[V × D]` | `[H × D]`，H << V |
| 内存 (D=4096, V=50K) | 200 MB | **16 MB**（trit 打包） |
| 查表 | 1 次大表读 | K 次小桶读（可并行） |
| Trit 打包后 | 200 MB | **4 MB** |

**Q1 已锁定为方案 1**（K=3-4 个桶拼出 token）。

### 2.2 数学公式

```
令 hash_k(id) = (id · salt_k) mod H,    salt_k 是预生成随机种子

x_t[d] = sum_{k=1..K} B[hash_k(token_id), d]
```

- `B[h, d]` 是桶池参数，训练时学习
- 输出 `x_t ∈ {-K, ..., +K}^D`（小整数向量）

---

## 3. Q2 · "输入"与"记忆"的概率  ✅ 已锁定（Q2-A + Sum 通道）

### 3.1 核心洞察

**单一通道不够**——必须有多个**互补**的状态通道：

| 通道 | 类型 | 作用 |
|------|------|------|
| `h` (Q2-A 主通道) | EMA 有界 `[H_VAL]` | **近期/位置敏感**任务 |
| `s` (Sum 通道) | 累加 有界 `[S_MAX]` | **全局/顺序无关**任务 |

### 3.2 Q2-A 主通道（EMA）

**更新公式**：
```
h_new[d] = clamp(round(α[d]·h_old[d] + (1-α[d])·x_t[d]), ±H_VAL)
```

- `h ∈ [-H_VAL, +H_VAL]^D`，**int8** 存储（H_VAL=4 时）
- `α ∈ [0, 1]^D`，**per-dim 学习的衰减率**
- 每步 O(D) ops，**纯整数运算**

**擅长**：
- "最后一个 token"（α 小）
- "最近 token 趋势"（α 中）
- 部分顺序敏感任务

**不擅长**：
- 全局聚合（顺序无关）：EMA 权重对位置倾斜，**做不到真正的 sum/avg**

### 3.3 Sum 通道（关键补全）✅ 实验验证

**更新公式**：
```
s_new[d] = clamp(s_old[d] + x_t[d], ±S_MAX)
```

- `s ∈ [-S_MAX, +S_MAX]^D`，**int16** 存储
- **无衰减，无乘法**，纯加法
- S_MAX 由序列长度决定（N × |x_max|）

**为什么必须**：
- `h` 通道做 EMA，权重对位置倾斜（α=0.9 时最近 token 权重是第 1 个的 5×）
- α=0.99 时权重接近均匀但信号极弱（h_max ≈ 0.5）
- **`sum` 是唯一天生顺序无关的算子**

**实验验证**（多数任务）：
```
D=4096, 16-token 序列, 多数 A vs 多数 B
Q2-A only (α=0.99):   66% 卡住
Q2-A + Sum:           100% @ epoch 1
```

### 3.4 两个通道的状态大小

```
h: D bytes (int8)         4 KB @ D=4096
s: 2D bytes (int16)       8 KB @ D=4096
x_t: D bytes (int8)       4 KB @ D=4096
α: 4D bytes (float)      16 KB @ D=4096

总计：32 KB 工作集 → 完全在 L1 缓存
```

---

## 4. Q3 · "数据与数据之间"的概率

### 4.1 自研设计 · 三层互补

| 层 | 机制 | 作用 | 状态 |
|----|------|------|------|
| 局部 | 短卷积 k=3 或 5 | 邻域语法 | 待实现 |
| 近期 | Q2-A 的 `h` 通道 | 末段趋势 | ✅ 验证 |
| 全局 | Q2-A 的 `s` 通道 | 累加聚合 | ✅ 验证 |

### 4.2 Sum 通道的实证（**关键实验**）

**Task A**：`count_a == 8`（s=0） vs `count_a ∈ {4, 12}`（s=±24）

| 头类型 | 准确率 | 备注 |
|--------|--------|------|
| 线性头 | 87-90% | 卡住：线性头不能区分 s=0 和 s=+24 |
| **多项式头 (含 s²)** | **100%** | s²=0 vs s²=576 干净分开 |

**Task B**：XOR(majority, last_token)

| 头类型 | 准确率 | 备注 |
|--------|--------|------|
| 线性头 | 46-53% | 卡住：XOR 非线性 |
| **多项式头 (含 h·s)** | **100%** | 同号 → label 0，异号 → label 1 |

---

## 5. Q4 · Output Head ✅ 已锁定（多项式头）

### 5.1 设计 · 多项式特征头

**输入**：`h ∈ ℤ^D`, `s ∈ ℤ^D`（两个通道的最终状态）

**特征**：
```
[h, s, s², h·s]      共 4 个 D 维特征
```

**输出**：
```
logit = w_h·h + w_s·s + w_s2·s² + w_hs·(h·s) + b
```

### 5.2 为什么多项式而不是 MLP？

| | 多项式头 | MLP (hidden=16) |
|--|--|--|
| 额外参数 | 2D | 32D |
| 参数量 | 8K | 131K |
| 表达力 | 限定二次形式 | 任意 |
| 适用于 | 多数聚合 + 简单非线性 | 复杂组合 |

**多项式对 sum 通道的天然适配**：s² 自动检测"sum 接近 0"。

### 5.3 全任务能力表

| 任务 | 通道 | 特征 | α (学到) | 准确率 |
|------|------|------|---------|--------|
| 多数 (majority) | s | s | 0.99 | 100% |
| 精确数 count=8 | s | **s²** | 0.99 | 100% |
| 最后 token | h | h | 0.05 | 100% |
| XOR(maj, last) | **s + h** | **h·s** | **0.05** | 100% |

**关键**：模型自己学到 α——同一架构既能"长期聚合"又能"短期位置"。

---

## 6. 终极蓝图 · v0.2 单层结构

```
输入 token_id
     ↓ Q1 哈希桶池 (16 MB 参数)
x_t ∈ {-K, ..., +K}^D           4 KB
     ↓
┌──────────────────────────────────────────┐
│  Q2-A 状态递推                              │
│                                           │
│  h_new[d] = round(α[d]·h[d] + (1-α[d])·x_t[d])  │
│  s_new[d] = s[d] + x_t[d]                 │
│                                           │
│  h: int8  [-4, +4]^D     4 KB              │
│  s: int16 [-64, +64]^D   8 KB              │
│  α: float [0, 1]^D       16 KB             │
└──────────────────────────────────────────┘
     ↓
┌──────────────────────────────────────────┐
│  多项式头                                  │
│  logit = w_h·h + w_s·s + w_s2·s² + w_hs·(h·s) + b │
└──────────────────────────────────────────┘
```

**单层参数 + 状态：**
- 桶池 Q1：16 MB（共享于所有层）
- α (per-layer)：16 KB
- head (per-layer)：16 KB（4 × D × 1 byte trit）
- 状态 h + s：12 KB（per-token）
- **总计：~16 MB 参数 + 12 KB per-token 状态**

---

## 7. 与现有工作的对齐

| 工作 | 对齐到自研蓝图 |
|------|----------------|
| **Mamba**（SSM） | Q2 状态递推（h 通道），但**没 Sum 通道**——Mamba 没法严格顺序无关聚合 |
| **RWKV**（WKV） | Q2 标量门控（α），但也没 Sum 通道 |
| **LFM2**（混合架构） | Q3 短卷积 + 主干递推——我们用 Sum 通道替代其长程依赖 |
| **低比特量化**（GPTQ / AWQ） | Q1 词表量化、Q4 head 量化 |
| **GLU / SwiGLU**（激活函数） | **不需要**——h 通道靠 α 门控自然实现非线性 |

**自研蓝图的独特点**：
- **Sum 通道**——比 Mamba/RWKV 多一个**严格顺序无关**的聚合器
- **多项式头**——比线性头多 2 个二次特征，参数只多 2D
- **per-dim 自适应 α**——比 RWKV 的标量衰减更灵活

---

## 8. 实测性能（D=4096, 单核, -O2 -march=native）

### 8.1 单步延迟

| 配置 | µs/step | ns/dim |
|------|---------|--------|
| Q2-A only | 16.49 | 4.02 |
| Q2-A + Sum | 21.34 | 5.21 |
| **Q2-A + Sum + Poly** | **20.60** | **5.03** |

### 8.2 吞吐量与扩展

```
D=4096, SEQ=1K:   22 ms/seq  →  46K tokens/s
D=4096, SEQ=4K:   85 ms/seq  →  47K tokens/s   ← 长度无关
D=8192, SEQ=1K:   54 ms/seq  →  19K tokens/s
```

### 8.3 内存

```
D=4096:  96 KB 工作集   (L2 友好)
D=8192: 192 KB 工作集   (L2 友好)
```

### 8.4 vs Transformer 单层

| | Transformer (D=4096) | Q2-A + Sum + Poly |
|--|--|--|
| Per-layer per-token | ~9 ms | **20 µs** |
| 倍数 | 1× | **~450× 更快** |

### 8.5 27B 类模型估算（32 层 × D=4096）

- 单 token 单线程：**0.64 ms** → ~1,500 tokens/s
- 32 路层并行：**0.02 ms** → ~50,000 tokens/s
- 每 token 状态：**12 KB**（vs Transformer 32 MB+ KV cache）
- 内存效率：**~2,500×** 优于 Transformer

---

## 9. 模型能力矩阵（已验证）

| 任务类型 | 例子 | 准确率 | 通道 |
|---------|------|--------|------|
| 近期位置 | 预测最后 token | 100% | h |
| 全局聚合 | 多数 (majority) | 100% | s |
| 精确数量 | count_a = 8 | 100% | s + s² |
| 跨通道组合 | XOR(maj, last) | 100% | h + s + h·s |

---

## 10. 待办与下一步

### 已完成 ✅
- [x] Q1 哈希桶池设计 + 实现
- [x] Q2-A 公式 + 测试
- [x] **Sum 通道**——突破聚合瓶颈
- [x] **多项式头**——解决非线性分类
- [x] 性能基准（per-step, scaling）
- [x] vs Transformer 对比
- [x] 架构文档 v0.2 整理

### 待完成
- [ ] **SIMD (AVX2) 优化**——目标 4× 加速比
- [ ] **多层级联**——32 层 27B-class 端到端测试
- [ ] **更复杂聚合任务**——OR-of-AND, parity, max-pool
- [ ] Q3 局部卷积与 Q2-A 串联/并联
- [ ] 实际长文本训练（Q1 + Q2-A 串起来）

---

## 11. 变更记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 初稿 | 建立四问四答框架与终极蓝图，尚未填公式 |
| v0.2 | 实测版 | + Q1 哈希桶池锁定，+ Q2-A + Sum 双通道，+ 多项式头，+ 实测数据 |