# Taovm 标准流程框架

原生 C++/CUDA 非 Transformer 三值双状态语言模型的训练流水线。
对应标准流程：**① 数据工程 → ② 分词器 → ③ 预训练 → ④ 退火 → ⑤ SFT → ⑥ GRPO → ⑦ 评测 → ⑧ 导出**

## 使用

```bash
cd pipeline
python -m taovm status          # 全部阶段状态
python -m taovm verify          # 依赖图自洽性检查
python -m taovm digest          # 各阶段产物 sha256（复现记录）
python -m taovm run 2_pretrain  # 执行阶段（先门控，后登记）
```

## 设计约定

1. **单一真源** —— `config.json` 是唯一配置来源。模型维度、阶段状态、产物路径都在此。
2. **配置即校验** —— `taovm/config.py` 在加载时拒绝违反架构硬约束的配置：
   - `m == d`（残差 `add` 的形状要求）
   - `d <= 1024`（`dual_state_autograd.cuh` 的 `dv>1024` 守卫）
   - `dk <= d`、`vocab >= FIRST_MERGE`
3. **不可跳步** —— 上游产物缺失则下游拒绝执行。这是为了防止"框架未定型就烧机时"。
4. **产物摘要** —— 每阶段完成后登记 sha256 到 `manifest.json`，支持陈旧检测。
5. **阶段脚本隔离** —— 每个阶段一个 `stages/<id>.py`，由框架调度，可单独重跑。

## 特殊 token 布局

```
0..255     原始字节
256..262   控制 token: BOS USER ASSISTANT TURN_END EOS THINK THINK_END
263..      BPE 合并 token（FIRST_MERGE = 263）
```

**注意**：新增控制 token 必须顺延 `FIRST_MERGE`，且会导致全部已分词分片作废。
旧分词器从 261 起，v2 从 263 起。

## 推理模式契约

```
U <问题>
A 思考 <逐步推理>答案：<最终答案>
```

- 推理语料：`data/p1_reason.txt`（202,652 篇）
- 训练/测试严格隔离：`data/grpo_train.jsonl`(202,652) / `data/grpo_test.jsonl`(3,104)，ID 与题干重叠均为 0
- `THINK=261` / `THINK_END=262` 已在分词器预留；当前用文本标记 `思考`/`答案：`

## 学习率约定

- `TAO_LR`：基础学习率（默认 1e-3，历史训练误用 5e-05，偏低 10–20 倍）
- 退火：`TAO_LR_DECAY_START` / `TAO_LR_DECAY_STEPS` / `TAO_LR_MIN`
- **`TAO_LR_DECAY_STEPS` 不得为 0** —— 会导致 `cos(NaN)`
- 收敛门控：`TAO_CONVERGE`，按分片训到收敛再进下一片（流式分片机制）

## 硬件约束（RTX 4070 Laptop 8GB）

实测吞吐（层数固定 2）。**显存余量是第一约束，比批大小本身更重要**：

| d | 参数 | slots × width | 位置/秒 |
|---|---|---|---|
| 3200 | **119.03M** | 16 × 3 | 125 |
| 3200 | 119.03M | 8 × 8 | 53（已触发 WDDM 驱逐） |
| 2048 | 60.85M | 16 × 8 | 435 |
| 2048 | 60.85M | 8 × 8 | 282 |
| 1024 | 23.62M | 16 × 16 | 1,790 |

d=3200 的显存模型（5 个实测点全部吻合到 ±25 MiB）：

```
峰值显存(MiB) ≈ 84.5 · slots · width + 40 · slots + 2121
```

**峰值一旦逼近 8188 MiB，WDDM 会在每次提交时做分配驱逐**：`ms_graph` 从约 1,000 ms
涨到约 5,000 ms，**所有阶段均匀慢 3 倍**（实测 8×8 峰值 7776 MiB ⇒ 53 位置/秒，
16×3 峰值 6817 MiB ⇒ 125 位置/秒，小而快）。按上式计算 16×4 是 8169 MiB，正落在
悬崖上，实测确实挂死。批形状可用 `TAO_PIPE_SLOTS` / `TAO_PIPE_WIDTH` 覆盖。

GEMM 的 batch 等于 `slots`，`width` 只是串行 BPTT 长度，所以「大 slots + 小 width」
更划算；但宽 256 那种配置（4096 单位）在任何 d 下都跑不起来。

## 权重驻内存

**前向权重驻显存，优化器状态驻主机 RAM。** 这不是风格选择，而是实测的物理约束：

| 权重位置 | t/步（d=1024） | 吞吐 | GPU 利用率 |
|---|---|---|---|
| 全部主机映射 | 15.36s | 132 tok/s | 94.4% |
| 全部显存 | 1.41s | 1,439 tok/s | 65.6% |

把**前向权重**放进主机内存会让每个 GEMM 经 PCIe 随机读权重，慢 10.9 倍 ——
GPU 利用率反而更高（94%），因为流处理器在等内存。因此：

- **前向传播读到的有效权重**（`embedding`/`vocab.bias`/各层矩阵）驻显存；
- **优化器状态**（`master`/`moment`/`variance`，共 3x 参数量）用
  `cudaHostAlloc(Mapped)` 驻主机 RAM，内核直接读写，**无每步 D2H/H2D 往返**。

每步只被 AdamW 顺序访问一次的优化器状态不构成带宽瓶颈，而它占 3 倍参数显存：
119M 参数下省下约 1.4 GB，正是本配置能在 8 GB 显卡上跑起来的原因。
