# Changelog

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
