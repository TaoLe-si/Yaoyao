# 13 · 2 层泛化 · L4 轮次状态与断点

> 本文是**断点续跑**用的状态快照。运行环境（`bash` / `glob` / `grep` / `run_code`）在
> 2026-09-11 本轮中途整体失效，错误为：
> `bash spawn failed: Error: subprocess-local: Windows Job runner exited with exit code 0 before proving its managed range empty`
> 子代理同样无法执行命令，因此**导出 / 编译 / 训练 / 评测全部无法运行**。

---

## 一、已完成（均已落盘，可验证）

### 1.1 架构固化：2 层

| 项 | 位置 | 值 |
|---|---|---|
| 新训练默认层数 | `src/dual_state_config.hpp:7` | `layers=2` |
| 权重共享 | `src/greedy_pipeline_grouped_model.hpp` `groupSource()` | `n==2 → {0,0}`（1 Full + 1 Reuse） |
| 旧 8 层 checkpoint 解码 | `applyDefaultArch()` | 未设 `TAO_LAYER_LIMIT` 时 `layer_limit=2` |
| 基准不再清深度 | `src/benchmark_noffn_s3.cpp:109` | 原无条件 `set_layer_limit(0)` 已删除 |
| 配置自证 | `config_dump()` | 打印 `layers=2/8` |

**实测**（旧 8 层 checkpoint `build/noffn_fresh/step_1512/final.dsb`，8 线程，自旋池）：

| 配置 | tps | µs/token | checksum |
|---|---:|---:|---:|
| **默认（=2 层）** | **8 522** | **117.3** | 5465 |
| 显式 `TAO_LAYER_LIMIT=2` | 8 563 | 116.8 | 5465（与默认一致 ✓） |
| 显式 `TAO_LAYER_LIMIT=8` | 4 184 | 239.0 | 278 |

### 1.2 旧架构路径删除

`src/cpu_pipeline_rows.hpp` 重写 + `src/greedy_pipeline_grouped_model.hpp` 清理，已删除：
float 权重（`FLOAT_W`）、AVX-512 浮点核（`AVX512`）、多累加器（`MACC`）、
2-bit 位平面（`BITPLANE`）、`KFOLD`、`TAO_ARCH_LEGACY` 分支。

**验证**：`scripts\build_s3.bat` exit=0；`scripts\build_v01.bat` exit=0（解码器 + 训练器均通过）。
残留：`dumpWeights()` 一处 `C4477`（`%6u` 收 `size_t`）警告，未修。

### 1.3 泛化基线（2 层，语料 L2）

语料 `data/noffn_l2/conversations.txt`：**2 133 篇 / 90 225 positions / 32 855 targets**。
训练 200 步（slots=32/width=32，lr 0.001），训练 NLL 15.8 → **0.09**。

交叉注入绑定率（`build/diag_binding_rate.exe`）：

| 检查点 | 名字 | 颜色 | **两者都对** |
|---|---:|---:|---:|
| step 100 | 2/18 | 4/18 | 0/18 |
| step 150 | 13/18 | 12/18 | 9/18 |
| **step 200** | **16/18** | **16/18** | **15/18** |
| 对照 8 层 probe3 @100 | 18/18 | 10/18 | 10/18 |

- 语料内名字：step 200 **12/12 两者都对**（8 层是 7/12）→ **2 层绑定能力不弱于 8 层**
- 语料外名字：3/6 两者都对

**但 held-out `eval_probe3.mjs` 在 step 100/150/200 全部 0/20。** 手工复现的两种病症：

1. **退化性重复**：`记记住了，你，你叫小树树树树树树…`（连发到 64 token 截断）、
   `小丽，橙色色。`、`记住了，你叫阿强强强强…`
2. **复制类任务为 0**：`请重复这句话：小王在看书。` → `记住了。`；`按顺序复述：红色、蓝色、绿色。` → `记住了，你叫阿强强强…`

### 1.4 语料 L4（已生成，未训练）

`data/noffn_l4/conversations.txt`：**18 038 篇 / 902 134 字符 / 12 541 个唯一用户话轮**
（100 250 行）。组成：

| 类别 | 篇数 |
|---|---:|
| 双槽绑定（4 种问法） | 7 000 |
| 双槽绑定（先问喜好，考第二槽位） | 2 500 |
| 列表复制 | 3 500 |
| 原样复制 | 1 500 |
| 算术 | 2 500 |
| 静态事实 | 720 |
| 话题承接 | 400 |

- 姓氏 200 × 名字 80 组合出名字；喜好 74 项
- **held-out 名字已排除在训练之外**：马超 / 郭静 / 孙丽 / 小林 / 小马 / 小孙 / 刘洋 / 小树
- `src/train_noffn_probe.cu:49` 文档上限 **4 096 → 65 536**（否则 18 038 篇会被拒）

### 1.5 一键脚本

`scripts/run_l4.bat [UPDATES]`（默认 400）：导出 → 重编 probe → 训练 → 绑定率 + held-out 评测。

---

## 二、待执行（环境恢复后立即跑）

```bat
cd /d D:\TaoVm
scripts\run_l4.bat 400
```

分解：

1. `build\export_noffn_probe.exe data\noffn_l4\conversations.txt build\noffn_l4\train.bin build\formal_tokenizer.bbp`
   （需先 `mkdir build\noffn_l4`；导出器拒绝覆盖已有产物）
2. `scripts\build_noffn_probe.bat`（nvcc，`Config.layers=2` 已生效）
3. `build\train_noffn_probe.exe build\noffn_l4\train.bin build\formal_tokenizer.bbp build\noffn_l4_run 400`
   （训练 >60 s，**必须后台运行**；日志写在 `build\noffn_l4_run.log`）
4. `build\diag_binding_rate.exe build\noffn_l4_run\step_N\final.dsb`
5. `node scripts\eval_probe3.mjs L4=build\noffn_l4_run\final.dsb`

---

## 三、判据（预登记）

在**匹配训练 NLL** 的前提下比较 L2（2 133 篇）与 L4（18 038 篇）：

| 指标 | L2 @NLL0.09 | L4 目标 |
|---|---|---|
| 交叉注入「两者都对」 | 15/18 | **≥ 15/18**（不得回退） |
| held-out 核心集 | 0/20 | **> 0/20**（首要） |
| 退化性重复 | 出现 | **消失或显著减少** |
| 复制类（list+repeat） | 0/8 | **≥ 4/8** |

**解读规则**：若 L4 在匹配 NLL 下 held-out 仍为 0/20 且退化重复依旧，则
「数据规模问题」的判断被证伪，需转向解码侧（重复惩罚 / no-repeat n-gram）
或架构侧；若 held-out 上升，则按 doc 06 §五 C4 继续放大语料与验证器。

---

## 四、下一步候选（按性价比）

1. **解码侧重复抑制**（不需重训，最便宜）：`GreedyResidentSession::reply_select`
   已有 `select(i,next)` 回调；可在 logits 层加 `TAO_REPEAT_PENALTY` / `TAO_NOREPEAT_N`，
   由 `greedy_step` 的 `observe(r,value)` 钩子实现。
2. **语料再放大**：L4 的生成器是程序化的，可无成本推到 50 M token 量级（doc 06 C4 阶梯）。
3. **深度旋钮 C3**：训练期随机丢弃 Reuse 层，部署 2/1 层两档（本项目唯一同时提速与保质的旋钮）。
4. **两段式候选池输出头**：`buildHead2` / `set_head2` 脚手架已在
   `greedy_pipeline_grouped_model.hpp:401`，但 `greedy_head_observe` 的评测路径从未写。
