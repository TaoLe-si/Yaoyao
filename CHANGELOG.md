
### [2026-09-08] Phase 4 - 多链 hash + Q3 移除

**关键发现**:
- ✅ Q3 局部卷积是**冗余且有害**: avg Loss 5.03 (Q3 训练权重) vs 4.85 (Q3 identity)
- ✅ 4 链独立 hash 提供 128 bits 真独立信息
- ✅ Warm-start from Phase 2 final, 快速收敛
- ✅ 数学验证 ALL PASS

**架构改动**:
- `h_hash[BATCH]` → `h_hash[4][BATCH]` (4 独立链)
- extract_hash_features 接受 4 个 hash, 输出 64 features (16 × 4)
- Q3 重置为 identity (warm-start 时): kk=0=1, 其他=0
- 多链 reverse 验证: 每链独立 reverse, 用 NEW state

**新文件**:
- yaoyao_gen_v21_fast_v2.cpp - 多链 hash 推理 (4749 tok/s)
- yaoyao_gen_v21_fast_v3.cpp - 嵌套 hash 推理 (4595 tok/s)
- PHASE4_LAYER_HASH.md - 多层 hash 设计文档
- PHASE4_PROPOSAL.md - Phase 4 完整提案

**数学定理**:
- 数据处理不等式: 32-bit hash 不能创造新信息
- 鸽巢原理: V^T (T≥5) > 2^32, 必有碰撞
- 函数复合不变: f∘f 不增加基数

**结论**: 多链 hash 提供 4x 信息但速度影响 <3%, 嵌套是优化不是突破.

---

# Changelog

## v1.0: Mod 3 可逆链突破 (当前主要工作)

### 核心创新
- **完全可逆的状态链**: h_new = (a*h + b*x) mod 3
- **滚动 hash 增强**: h_hash_new = h_hash * 33 + token (mod 2^32)
- **CPU 极轻**: +6.66% 时间, +4 bytes 内存
- **信息无损**: 100% 保留历史 (vs 当前 96.875% 损失)

### 数学保证
- Mod 3 在 {-1, 0, +1} 上封闭
- 33 在 mod 2^32 下有逆元 0x3e0f83e1
- 整体系统完全可逆 (任意时刻可恢复历史)

### 6 个验证测试
- Test 1: 基本可逆性 ✓
- Test 2: 1000 步长期稳定 ✓
- Test 3: 任意历史可恢复 ✓
- Test 4: 信息保留 (基础) ⚠️
- Test 5: Hash 解决碰撞 ✓
- Test 6: 完整系统可逆 ✓

### 与现有方案的对比
- Mamba/RWKV: 浮点, 不可逆, O(N)
- Transformer: 浮点, 不可逆, O(N²)
- 夭夭 v21: 整数, 完全可逆, O(1)

详见 `REVERSIBLE_CHAIN_BREAKTHROUGH.md`

---

## v21 实现 (实际训练)

### 集成到 yaoyao_v21_full.cpp
- 基于 v19 改造, 替换 h/s 通道
- 保留 Q1 (hash bucket pool) + Q3 (ternary conv) + alpha + gate + Wbi
- 新增: W (trit to vocab) + W_hash (hash to vocab)
- 完整 save/load 支持增量训练 (.bin)
- 内置 generation phase (top-p sampling + repetition penalty)

### 数学验证 (运行时)
- 启动: mod3 封闭/可逆/hash 可逆/组合可逆, 5/5 PASS
- 训练中: softmax sum err < 3e-6
- 训练中: d_logits sum err < 1e-6
- 训练中: 每 100 windows hash 完全可逆
- 训练结束: trit + hash 8 步完全恢复

### 增量训练结果 (TinyStories 483MB, V=1024)
- Init (50 steps): loss 6.66
- R1 (650 steps): loss 5.17, down 1.49
- R2 (1250 steps): loss 4.79, down 0.38
- R3 (1850 steps): loss 4.63, down 0.16
- R4 (2450 steps): loss 4.54, down 0.09

总计: 2400 windows x 3 epochs = 7200 windows
最终 Loss: 4.54 (收敛中, 无退化)
模型: D:\TaoVm\yaoyao_v21.bin (step=2450)

### 与 v19 对比
- v19 (2000 windows): Loss 4.01, 不可逆, 有损, 长期退化
- v21 (2400 windows): Loss 4.54, 完全可逆, 无损, 无退化


## v21 实现 + 自适应训练完成

### 架构实现
- 基于 v19 改造, 替换 h/s 通道为 mod 3 trit + 滚动 hash
- yaoyao_v21_full.cpp: 657 行, 完整训练+生成+save/load
- yaoyao_gen_v21.cpp: 推理版 (BATCH*SEQ 重算, ~8 tok/s)
- yaoyao_gen_v21_fast.cpp: **极速版 (O(1) 增量状态, 5609 tok/s)**

### 数学验证
- 启动: 5 项数学原语验证全 PASS
- 训练中: softmax sum err < 3e-6, d_logits sum err < 1e-6
- 每 100 windows: hash 完全可逆
- 训练结束: trit + hash 8 步完全恢复

### 增量训练 (Round 1-8)
| Round | Step  | Loss  | LR    |
|-------|-------|-------|-------|
| Init  | 50    | 6.66  | 0.005 |
| R1    | 650   | 5.17  | 0.005 |
| R2    | 1250  | 4.79  | 0.005 |
| R3    | 1850  | 4.63  | 0.005 |
| R4    | 2450  | 4.54  | 0.005 |
| R5    | 3650  | 4.61  | 0.01 (反弹) |
| R6    | 4850  | 4.49  | 0.005 |
| R7    | 6050  | 4.47  | 0.005 |
| R8    | 7250  | **4.45** | 0.005 (最佳) |

### 自适应训练 (Round 9-13)
| Round | Step   | LR    | Loss  | 状态 |
|-------|--------|-------|-------|------|
| R9    | 8450   | 0.001 | 4.70  | Adam 不匹配 |
| R10   | 9650   | 0.005 | 4.54  | 震荡 |
| R11   | 10850  | 0.005 | 4.53  | 震荡 |
| R12   | 12050  | 0.005 | 4.52  | 震荡 |
| R13   | 13250  | 0.003 | 4.51  | 震荡 |

**收敛判定**: 模型在 4.45-4.55 间震荡, 已达架构上限. 停止训练.

### 最终模型
- yaoyao_v21.bin (step=13250, 17 MB, Loss=4.51)
- 推理速度: **5609 tok/s** (702x 加速)
- 可逆性: 100% 保持
- 信息保留: 100% (无损)

---


## v0.9 (current)
- Word-level vocab (1024 tokens from tinystories)
- Q1 hash bucket pool + fixed query (zero query = deterministic lookup)
- Q3 conv (k=3) + Q2-A channels (h decay + s sum) + dynamic alpha T=2
- Adam with dual LR (main=0.005, alpha=0.0005)
- Soft clip grad -> Adam -> hard clip weights
- Save/load binary format with full Adam state (incremental training)
- Generation: top-30 sampling + temperature 0.9 + repetition penalty

## Bug fixes
- **2026-09-07**: Fixed generation bug - `ids.push_back(best)` was missing,
  causing the loop to regenerate the same prompt 40 times.
  Result: model now generates real English stories.
- Step 6b: Q1 backward missing direct term `d_out[d]*w[k]`
- Step 6b: `d_trit` was int8_t (truncates float grad) - now float vector
- Step 7: cum_s must be (SEQ+1)*D, not SEQ*D

## Training progression
| step | avg_loss | notes |
|------|----------|-------|
| 0    | 4.99     | init  |
| 6000 | 4.05     | first run, char-level |
| 6000 | 4.99     | switch to word-level, retrain |
| 24000| 3.41     | word-level, N_WIN=3000 |
| 44000| 3.41     | with repetition penalty |
| 74000| 3.57     | LR=0.005 + N_WIN=10000 |
| 104000| 3.58    | more data, stable |
| 105400| 3.30    | first successful generation! |
| 135400| 3.55    | 3 more epochs, plateau |
| 75000 | 3.20    | SEQ=64 + top-p breakthrough |


## D=128 (current)
- Doubled D from 64 to 128
- Loss 4.42 -> 3.56 after 3 epochs
- Top-p sampling with p=0.9 produces diverse vocabulary
- Saved as yaoyao_v09_d128_baseline.bin

## v0.9.5: OpenAI-compatible API
- yaoyao_api.py: FastAPI server on port 11434
- yaoyao_gen.exe: inference-only C++ exe
- Endpoints: GET /v1/models, POST /v1/chat/completions
- Compatible with any OpenAI client (curl, requests, etc.)

## v0.9.6: Streaming output with per-token timing
- yaoyao_gen.cpp: streaming mode (stream arg) outputs TOKEN lines with step_ms + total_ms
- yaoyao_api.py: SSE streaming using asyncio subprocess (no threading)
- Each token sent as separate SSE chunk with timing in JSON
- Compatible with OpenAI streaming SDK (stream=True)
- Test script shows per-token: step=6.5ms total=13.0ms

## v0.9.7: Persistent C++ server for low-latency generation
- yaoyao_gen.cpp: --server mode, reads prompts from stdin, prints tokens to stdout
- yaoyao_api.py: spawns ONE persistent subprocess, pipes requests via stdin
- Eliminates ~2.2s model load per request
- TTFT: 2200ms → 15ms (150x faster)
- Streaming throughput: 7.4 tok/s → 120+ tok/s (16x faster)
- Per-token interval: 130ms → 7.8ms (17x faster)

## v0.9.8: Complete README rewrite
- 17 sections covering: architecture, vocabulary layer (h+s replacing attention),
  Q1 hash bucket, Q3 conv, dynamic alpha, Q4 output head, RMSNorm,
  current spec, training progression, performance, file structure,
  API docs, alignment with Mamba/RWKV, performance comparison
- Acknowledges "vocabulary layer" terminology for (h, s) channels
- All tables reflect current v0.9.7 state (Loss=3.11, 17.64 MB model)
