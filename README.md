# 夭夭 (Yaoyao) v21: CPU 原生大语言模型

## 🎯 项目定位

**夭夭** 是一个**完全跑在 CPU 上、无自注意力、无 KV cache**的小型语言模型.

核心创新:
- 🔁 **可逆链 (Reversible Chain)**: mod 3 trit 累积 + 滚动 hash, 数学严格可逆
- 💾 **O(1) 内存**: 状态大小固定, 与上下文长度无关
- 🚫 **无 Attention / 无 KV cache**: 用数学函数替代 softmax attention
- ⚡ **CPU 友好**: 5609+ tok/s 推理速度 (H=64 SwiGLU)

---

## 📐 架构 (Phase 4 多链 hash)

### 信息通道

|通道 | 编码内容 | 容量 | 数学性质 |
|------|---------|------|---------|
| **Trit (mod 3)** | 累积 trit 状态 | 3^128 ≈ 2^203 bits | 严格可逆 |
| **Hash (4 链独立)** | rolling hash 状态 | 4 × 32 = 128 bits | 严格可逆 (4 链独立可逆) |
| **Q1 bucket pool** | 词嵌入查询 | B=128, K=16 | 软注意力 |
| **Q3 conv** *(已移除)* | 局部 5-gram | ~~5 × 2 × 128 = 1.3K~~ | 不必要 |
| **Wbi bigram bias** | 双 token 跳转 | V × V = 1M | 学到的 |
| **SwiGLU pred head** | 隐藏层 | HIDDEN=192 | 标准 |

### 信息上限

```
trit bits    : 3^128 ≈ 2^203 bits (全局历史)
hash bits   : 4 × 32 = 128 bits (滚动状态,4 独立链)
Wbi         : 1024×1024 = 1M 参数 (双 token 模式)
总可学信息  : ~210 bits 状态 + 17M 权重
```

### Forward 流 (每 token)

```
1. Q1 lookup:    x = softmax(Q1.trits[id] · query) · trits
2. Q3 conv:      ~~跳过~~ (Phase 4 发现: Q3 有害)
3. alpha gate:   α = sigmoid(aW·x + ab)
4. gate gate:    g = sigmoid(gW·x + gb)
5. trit update:  h_trit[d] = mod3(h_trit[d] + Q1.trits[bucket][d])
6. hash update:  h_c = h_c * base_c + token + offset_c   (4 chains, c∈{0,1,2,3})
7. extract:      features[16×4] = bit_slice + murmur + jenkins + splitmix
8. SwiGLU:       hidden = silu(W_gate · state) * (W_up · state)
9. logits:       v = Wbi[prev,v] + W_sgl_out[v,:] · hidden
10. softmax:    p = softmax(v / T)
```

### Hash 链设计 (Phase 4)

```cpp
// 4 条独立 hash chains, 每个 32 bits
const hash_t HASH_BASES[4]   = {33, 37, 41, 43};    // 互素
const hash_t HASH_INV[4]     = {0x3e0f83e1, ...};    // mod 2^32 逆
const hash_t HASH_ADDS[4]    = {7, 11, 13, 17};

// Forward: 每链独立更新
h[0] = h[0] * 33 + token + 7;
h[1] = h[1] * 37 + token + 11;
h[2] = h[2] * 41 + token + + 13;
h[3] = h[3] * 43 + token + 17;

// Reverse: 严格可逆
h[c] = (h[c] - token - HASH_ADDS[c]) * HASH_INV[c];

// Extract: 4 种不同雪崩,每链提供 16 features
features[0..15]   = bit_slice(h[0]);
features[16..31]  = murmur3(h[1]);
features[32..47]  = jenkins(h[2]);
features[48..63]  = splitmix64(h[3]);
```

---

## 🏗️ 训练阶段演进

| Phase | 配置 | Step | Loss | 关键发现 |
|-------|------|------|------|---------|
| Phase 0 | D=128 H=16 linear | 13250 | 4.45 | baseline linear |
| Phase 1 | D=128 H=64 linear | 7250 | **4.05** ⭐ | H=64 大幅改善 |
| Phase 2 | D=128 H=64 SwiGLU | 21100 | 4.34 | SwiGLU 微涨 (更复杂) |
| Phase 3 | D=128 H=128 init | 200 | 5.51 | 中断 |
| **Phase 4** | **多链 hash** | 24800 | **4.61** | **Q3 有害, 移除** |

### 关键实验发现

**🔬 Q3 局部卷积是冗余且有害的**

对比实验 (Phase 4 + warm-start):
- Q3 训练权重 (Phase 2): avg Loss 5.03
- Q3 identity (no-op): **avg Loss 4.85** ← 更好!

**结论**: Q3 卷积 (K=5) 不仅没用, 反而引入噪声. trit + hash + alpha + gate 已经足够. **Q3 已从架构中移除**.

**🔬 多链 hash 提供真正独立信息**

4 条独立 hash chain (base 33/37/41/43) 提供 128 bits 真独立信息, 对比单链的 32 bits.

---

## 🧮 数学模型

### Mod 3 可逆性

```
trit_new = mod3(trit_old + embed)      ∈ {-1, 0, +1}
trit_old = mod3(trit_new - embed)      ← 严格逆运算
```

### Hash 可逆性

每条 chain:
```
h_new = h_old * base + token + offset   (mod 2^32)
h_old = (h_new - token - offset) * base^(-1)   (mod 2^32)
```

其中 base 与 2^32 互素, 所以逆元存在:
- 33⁻¹ = 0x3e0f83e1
- 37⁻¹ = 0x914c1bad
- 41⁻¹ = 0xc18f9c19
- 43⁻¹ = 0x2fa0be83

### 不可逆性证明 (定理)

**数据处理不等式**: 任何 32-bit hash 函数的输出 ∈ 2^32, 不能区分 V^T = 1024^T 种 token 序列 (T ≥ 5 时 2^1000 > 2^32).

**鸽巢原理**: 必有不同输入 → 同一 hash (碰撞不可避免).

**函数复合不变**: f: 2^32 → 2^32 的复合仍映射到 2^32. **递归 hash 不能创造新信息**.

---

## 🚀 性能

| 指标 | 数值 |
|------|------|
| 模型大小 | ~19 MB (D=128, NL=2, H=64) |
| 训练速度 | ~200 windows/min (BATCH=16, SEQ=64) |
| 推理速度 (slow) | ~8 tok/s |
| **推理速度 (fast)** | **5609 tok/s** (incremental state) |
| 内存占用 | O(1), 235 bits/position |

---

## 📁 代码结构

```
D:\TaoVm\
├── yaoyao_v21_full.cpp       # 训练 + 生成 (953 行)
├── yaoyao_gen_v21_fast.cpp   # 快速推理 (5609 tok/s)
├── yaoyao_gen_v21_fast_v2.cpp # 多链 hash 版 (4749 tok/s)
├── yaoyao_gen_v21_fast_v3.cpp # 嵌套 hash 版 (4595 tok/s)
├── analyze_v21.cpp           # 模型利用率分析器
├── yaoyao_v21.bin            # 当前模型
├── yaoyao_v21_*.bin          # 各阶段备份
├── tinystories_train.txt     # 472MB 训练数据
└── *.md                       # 文档
```

---

## 🎬 当前状态 (Phase 4)

- ✅ 数学验证: ALL PASS
- ✅ 4 链 hash + SwiGLU 训练运行中
- ✅ Q3 局部卷积已移除 (实验证明有害)
- ✅ Warm-start 从 Phase 2 final
- 🔄 训练 step 24800, Loss 4.76 (收敛中)
- 🎯 目标: Loss < 4.34 (Phase 2)

---

## 📚 文档

- [CHANGELOG.md](CHANGELOG.md) - 完整更新日志
- [cpu-native-llm-architecture.md](cpu-native-llm-architecture.md) - 架构详细说明
- [数学模型.md](数学模型.md) - 数学推导
- [measurements.md](measurements.md) - 性能基准
- [REVERSIBLE_CHAIN_BREAKTHROUGH.md](REVERSIBLE_CHAIN_BREAKTHROUGH.md) - 可逆链突破
- [PHASE4_LAYER_HASH.md](PHASE4_LAYER_HASH.md) - 多层 hash 设计
- [PHASE4_PROPOSAL.md](PHASE4_PROPOSAL.md) - Phase 4 完整提案
- [PHASE2_SUMMARY.md](PHASE2_SUMMARY.md) - Phase 2 SwiGLU 总结

---

## 🔬 信息论上限

```
trit bits: 3^128 ≈ 2^203 bits
hash bits: 128 bits (4 独立链)
总状态: ~235 bits/position
有效信息: ~85-130 bits (利用率 50-65%)

天花板 Loss (架构极限): ~3.0-3.3
当前可达: 4.5-4.7 (Phase 4 + Q3-off)
```

要突破 235 bits 上限, 必须:
1. 扩大 D (Trit bits)
2. 加宽 hash (64-bit hash)
3. 引入新信息源 (position, bigram, etc.)

---

## 🚧 GitHub

仓库: https://github.com/TaoLe-si/Yaoyao.git

每次重要更新自动推送.

---

## 📜 许可证

本项目为开源研究项目.

---

**作者**: Tao (2584300846@qq.com)
**项目启动**: 2025
**当前版本**: v21 Phase 4 (多链 hash + Q3 removed)
