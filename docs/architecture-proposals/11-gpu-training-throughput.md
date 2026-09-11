# 11 · GPU 训练吞吐：根因、修复与实测加速

> 本文记录 2026-09-11 对 `train_noffn_probe`（probe3 语料，8 层 d=512）的训练吞吐优化。
> 所有数字均为**【实测】**，同机同日、**串行**运行（并发会让 replay 时间翻倍以上，见 §7）。
> 语料指纹：`docs=969 positions=37736 targets=14908 sha256=654690f0…ba5`，未改动。

---

## 一、结论摘要

| 项 | 优化前 | 优化后 | 倍数 |
|---|---:|---:|---:|
| 每 position 耗时（profile_probe，单步） | 4.19 ms | **0.248 ms** | **16.9×** |
| 端到端吞吐（真实训练器，含建图/落盘） | 234 pos/s | **3635 pos/s** | **15.5×** |
| 遍历一遍语料（37 736 positions） | 161 s | **10.4 s** | 15.5× |
| 单个优化器步墙钟 | 5.52 s | 2.50 s | 2.2× |

**加速的全部来源都不是算法改动**：算子定义、张量集合、数值语义都没有变；前两项是**纯配置**（batch plan 的 `slots` / `width`），第三项是等价的**访存分块**（同一数学，FP 求和顺序不同）。

**但两者的贡献极不对称——本文初稿把它们混在同一张 A/B 表里，容易被读成"内核是主因"。2026-09-11 隔离实测（§九）的分解是：**

| 来源 | 贡献 |
|---|---:|
| batch plan 配置（`4/128` → 大 `slots`） | **≈11.9×** |
| slot 分块内核 | **≈1.15–1.28×** |

**主因是配置，不是内核。** 且分块内核要求 `slots ≥ 8 && slots % 8 == 0`，否则**静默回退**到 baseline 内核（此时与默认构建数值逐位相同）。

---

## 二、根因（实测定位）

`build/profile_probe.exe` 逐相位计时：

| 相位 | 4 slots / width 128 | 占比 |
|---|---:|---:|
| CUDA graph replay | 4583 ms | **99.5 %** |
| loss | 0.1 ms | ~0 % |
| optimizer update（含 host 读回） | 23.1 ms | 0.5 % |

所以瓶颈**不在**主机读回，**全在 batch graph 的一次 replay**。三个叠加原因：

1. **图的宽度按 128 捕获，而真实 plan 平均只有 30–48 个 timestep。**
   `PLAN timesteps=30 slots=4` → 占用率 **20.3 %**，即 3–4 倍的时间步是纯 padding。
   原因：`take_batch(cursor,width)` 里 `p.timesteps = max_k(work[k].end-begin)`，而每个 slot 每个 batch 只 `take` 一次（`pilot_slot_cursor` 取 `min(width, doc剩余)`），所以把 `width` 设大并不能填满，只是把 padding 拉长。
2. **只有 4 个序列 slot。** 每层浮点权重 6.7 MB、全模型 87 MB；`ds_batched_matvec` 对每个 slot 都把 `w` 完整读一遍。slots=4 时权重复用率极低。
3. **`ds_batched_matvec` / `ds_batch_dx` 没有 slot 分块。**
   在 32 slots 下实测 L2 流量约 **665 GB/s**（每 timestep 前向+反向要搬 5.6 GB），已经是访存墙。

---

## 三、修复一：batch plan 配置（不需要改代码）

`train_noffn_probe` 新增两个可选参数：

```
train_noffn_probe <TRAIN.bin> <TOKENIZER.bbp> <OUT_DIR> <UPDATES> [SLOTS WIDTH]
```

缺省 `SLOTS=4 WIDTH=128`，与历史行为完全一致。实测扫描（baseline 内核，单步）：

| slots | width | 占用率 | ms/position |
|---:|---:|---:|---:|
| 4 | 128 | 20.3 % | 4.269 |
| 4 | 64 | — | 2.06 |
| 4 | 48 | — | 2.09 |
| 8 | 48 | — | 1.07 |
| 16 | 48 | — | 0.552 |
| 32 | 48 | — | 0.434 |
| 32 | 32 | 78.7 % | 0.354–0.368 |
| 48 | 48 | 87.3 % | 0.332 |
| 64 | 24 | 87.8 % | 0.324 |
| 64 | 32 | — | 0.322 |
| 96 | 32 | — | 0.322 |
| 128 | 32 | — | 0.326 |

- **宽度**从 128 降到 64/48：4.269 → 2.06，**2.07×**（占用率 20 % → 88 %）。
- **槽数**从 4 增到 64（宽度≈24–32）：2.09 → 0.324，**6.5×**。
  增益超过线性，因为权重矩阵开始被 L2 复用：每 slot-step 成本从 1.394 ms（4 slots）降到 0.239 ms（64 slots）。

**推荐配置**：`slots=32 width=32`（平衡，峰值显存 2.95 GB）或 `slots=64 width=24`（最省时间，峰值 3.43 GB）。
`96/24` 更快一点（0.240）但每次更新吃 14 164 个 position，在 37 736 的探针语料上一遍只剩 2.7 步，优化粒度太粗，不推荐。

---

## 四、修复二：按 slot 分块的 GEMM 内核

新增（`-DTAO_SLOT_TILED_GEMM` 开启，默认关闭）：

- `src/gpu_batched_matvec_slots.cuh`：`ds_bmv_tiled_kernel<TILE>` — 一个 warp 负责一行、寄存器里同时累加 `TILE` 个 slot，权重行只读一次；`x` 的 slot 组先搬进 shared memory。
- `src/gpu_batched_backward.cuh`：`ds_dx_tiled_kernel<TILE,COLS,CHUNK>` — 一个 block 覆盖 `TILE` 个 slot 与一个列分块，权重每个 slot 组只读一次；**行方向再切 `CHUNK` 段**（不够并行会变慢，见下）。
- `TILE` 默认 8。

实测（3 次重复，均值）：

| 配置 | baseline 内核 | 分块内核 | 加速 |
|---|---:|---:|---:|
| 32 / 32 | 0.368 | **0.301** | 1.221× |
| 64 / 24 | 0.324 | **0.248** | 1.305× |
| 4 / 128 | 4.19 | 4.19（`TILE>slots`，自动回退） | 1.000× |

**踩过的坑**：第一版 `ds_dx_tiled_kernel` 不做行切分，grid 只有 `(cols/32)×(slots/8)` 个 block，
在 32 slots 下并行度不足，**反而慢 12 %**（0.368 → 0.414，3 次重复确认）。加上 `blockIdx.z` 行切分（`CHUNK=128`，多段用 `atomicAdd` 合并）后才转为全面领先。
`TILE` 也不是越大越好：16 无增益（1.008×），32 反而退化到 0.829×（寄存器/占用率压力）。

---

## 五、端到端 A/B（真实训练器，同一语料与哈希）

```
A: train_noffn_probe2.exe    <corpus> <tok> build/fin_base 5  4 128     # 现状
B: train_noffn_probe_fast.exe <corpus> <tok> build/fin_fast 12 64 24    # 加速 + 分块
```

| | A（4/128） | B（64/24） |
|---|---:|---:|
| updates | 5 | 12 |
| positions | 6 474 | 109 028 |
| 墙钟 | 27.6 s | 30.0 s |
| **吞吐** | **234 pos/s** | **3635 pos/s（15.5×）** |
| 每优化器步 | 5.52 s | 2.50 s |
| NLL 轨迹 | 16.017 → 15.569 → 14.470 → 13.275 → 11.819 | 16.039 → … → 8.007 → 7.645 → 7.227 |

两端 `PROBE_START` 的 `sha256` / `tokenizer` 一致，语料与分词器未变。

> **注意：本表的 A/B 同时改变了「配置」与「内核」两个变量，因此不能用来归因。** 隔离分解见 §九：配置 ≈11.9×，内核仅 ≈1.15–1.28×。

---

## 六、必须一起说明的代价与边界

1. **这是吞吐胜利，不是免费的收敛胜利。** 槽数变大后每次更新消耗的 position 数从 1 143 涨到 9 546（**5.6×**），
   一遍语料从 ~33 步变成 ~6 步。**达到同样 NLL 所需的优化器步数会变**，需要重新调 lr / 步数。
   换言之：per-position 吞吐提升 15–17×，**per-optimizer-step 只快 2.2×**。
2. **分块内核改变了 FP 求和顺序。** 数学等价，但梯度不再逐比特相同，因而**不能与默认构建的 run 逐比特对比**。
   默认构建保持逐字节不变；加速版是独立可执行文件（`build/train_noffn_probe_fast.exe`）、并在 `PROBE_START` 里带 `kernels=slot-tiled` 标识。
3. **显存随 `slots` 线性增长**（每 slot 每层要额外的激活与梯度缓冲）：
   4/128 峰值 3.10 GB、32/32 2.95 GB、64/24 3.43 GB、96/24 4.26 GB（8 GB 卡）。
4. 并发跑测量会严重污染结果（观察到 replay 12572.8 ms vs 干净 4583 ms）。**本文件所有数字都是串行测的。**

---

## 七、仍未吃到的优化（按预期收益排序）

| 方向 | 依据 | 状态 |
|---|---|---|
| `ds_batch_dw` 未分块 | 每线程为 1 个输出要读 `2×slots` 个 float（32× 放大），量级与未分块前向相当 | 未做 |
| 未监督位置跳过输出头 | 头占每 token MAC 的 38.6 %；probe3 上约 23 % 的位置无监督目标 | 未做 |
| `update()` / `project()` 的 host 读回 | 24–30 ms/update；在 64/24 下已只占 ~1 %，随槽数增大会重新变重要 | 未做 |
| H2R 的梯度检查点 | 见 10 号文档 §11，矩阵状态使激活显存涨 64× | 未做 |

---

## 八、复现

```bat
scripts\build_profile_probe.bat              :: 建 build/profile_probe.exe（baseline）
scripts\build_profile_tiled.bat              :: 建 build/profile_tiled.exe（-DTAO_SLOT_TILED_GEMM）
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 32 32 1
scripts\build_noffn_probe_fast.bat           :: 建 build/train_noffn_probe_fast.exe
build\train_noffn_probe_fast.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\out 12 64 24
```

`profile_probe` 用法：`profile_probe <TRAIN.bin> <TOKENIZER.bbp> <SLOTS> <WIDTH> [UPDATES]`，
会打印 `PLAN … 占用率`、`VRAM after-capture/after-update`、以及每步的 `replay / loss / update / TOTAL ms / ms-per-position`。

---

## 九、转正：加速版成为默认（2026-09-11）

### 9.1 隔离测量：内核 vs 配置

在同一语料（probe3）、同一优化器配置下把两个变量分开测：

| 配置 | 内核 | ms/position | 说明 |
|---|---|---:|---|
| 4/128 | baseline | 5.00 | 旧默认 |
| 4/128 | tiled | 5.37 | **未生效**：`slots=4 < 8` → 回退 |
| 8/128 | baseline | 2.530 | |
| 8/128 | tiled | 2.714 | 生效但**更慢**（block 数太少） |
| 32/32 | baseline | 0.492 | |
| 32/32 | tiled | **0.425** | |

**结论：**

1. 吞吐主要由 `slots` 驱动（4 → 32 带来约 10×），内核只在其上再加 **1.15–1.28×**。
2. `slots < 8` 或 `slots % 8 != 0` 时 `launch_bmv_tiled` / `launch_dx_tiled` 返回 false，**静默回退**到 baseline 内核。
   4/128 下"加速版"与默认构建**逐位相同**（六步 NLL 差 0.00e+0）——这就是"只翻内核宏在默认配置下等于没做"的原因。
3. 8/128 下分块内核反而更慢：`TILE=8` 时 `grid.y = slots/TILE = 1`，并行块数不足。

### 9.2 默认值变更

| 位置 | 旧 | 新 |
|---|---|---|
| `batch_tape.cuh`、`gpu_batched_backward.cuh`、`train_noffn_probe.cu` | `TAO_SLOT_TILED_GEMM` 显式开启分块 | **默认开启**；`TAO_BASELINE_GEMM` 作退出口 |
| `train_noffn_probe.cu` | `slots=4, width=128` | **`slots=32, width=32`** |
| `train_yaoyao_graph_gpuval.cu` | `slots=4, width=256` | **`slots=32, width=32`** |
| `train_noffn_fresh.cu` 策略串 | `graph4x8x256` | `graph32x8x32-slot-tiled` |

**生产路径每步数据量不变**：旧 `8 轮 × 4 slots × 256 width = 8192`，新 `8 × 32 × 32 = 8192`。
因此原有的 lr / 步数调度**不需要重调**——这是选 `32/32` 而非 `64/24` 的主要理由之一（`64/24` 更快，0.323 ms/pos，但它改变了每步数据量）。

**一个必须知道的语义变化**：每条并行序列的展开长度从 **256 降到 32**（8 GB 显存装不下"大 slots × 长展开"）。
每步总 token 量不变，但单序列变短。保守替代是 `16/64`（同样 1024/轮，约 0.81 ms/pos，仍有约 3× 加速）。

### 9.3 实测验证

重建 `yaoyao_train_v01.exe`、`train_delta_fast.exe`、`train_noffn_probe2.exe`、`train_noffn_probe_fast.exe`（全部 exit=0），
用**不带任何参数**的默认构建跑：

```
PROBE_START ... slots=32 width=32 kernels=slot-tiled
0.425 ms/position   vs   旧默认 5.00 ms/position   →  11.8×
```

### 9.4 旧权重作废

默认值变更改变了 batch plan 的抽样形状，`train_noffn_fresh.cu` 的策略串亦已修改，
**旧检查点的 identity 与之不再匹配**，按 I5 不会被静默复用。生产侧一律重新训练。

### 9.5 已知未修的问题（明确记录，本次不修）

`ds_dx_tiled_kernel` 在 `rows > CHUNK(=128)` 时用 `atomicAdd` 跨 `blockIdx.z` 合并（`gpu_batched_backward.cuh:26`），
**浮点求和顺序不确定 → 同一命令两次运行的 NLL 不逐位相同**（实测同一步 0.5305 vs 0.5384）。

因为旧权重已作废、可复现性的锚点转移到了新产物，本次转正**不修**。
若将来需要逐位复现（例如做严格 A/B 对比），必须改用 `-DTAO_BASELINE_GEMM` 构建，
或把该内核改成"每个 z 写独立部分和 + 定序归约"（代码库中 `batch_rms_backward` + `batch_bias_grad` 已是这个模式）。
