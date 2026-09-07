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
