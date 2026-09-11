# 24 · R4 算子统一：训练器与解码器改用同一激活

> 本文记录一个**违反 R4 铁律**的真实缺陷及其修复。
> 该缺陷此前一直存在，直到用 `decode_diag` 做 A/B 才暴露。

---

## 一、缺陷

R4 要求「训练/推理同算子」。实测**不满足**：

| 侧 | 文件 | 状态更新用的激活 |
|---|---|---|
| **训练器（GPU）** | `dual_state_cuda_resident.cuh` | 精确 `ds_sigmoid`（`expf`）+ `tanhf` |
| **解码器（CPU）** | `cpu_fast_activation.hpp` | 三阶 Padé 近似 |

解码器 `applyDefaultArch()` 默认 `fast_act_=true`，走 `fast_gated_update`：

```
tanh(x)    ≈ x(27 + x²) / (27 + 9x²)，|x| > 3 夹紧 ±1
sigmoid(x) = 0.5·(1 + tanh(x))
```

`cpu_fast_activation.hpp` 自述该近似在 `[-6,6]` 上**最大绝对误差 0.0235**。

**后果**：所有历史检查点都是用精确激活训练的，却用 Padé 近似解码。
在 2 层 × 512 维 × 每个 token 上累积，输出必然偏离训练分布。

### 实测证据

`decode_diag` 在 L6 step_500 上做 A/B（仅切换 `TAO_FAST_ACT`）：

| 提示 | `TAO_FAST_ACT=1`（解码器默认） | `TAO_FAST_ACT=0`（与旧训练器一致） |
|---|---|---|
| `只输出数字：7×8等于多少？` | `12` | `144` |
| `只输出数字：11×4等于多少？` | `144` | `144` |
| `请按顺序输出：铅笔、草莓、桃子` | `铅笔、草莓、草莓、桃子、…` | `铅笔、草莓、草莓、桃子、草莓、…` |

**输出确实不同** → 确认是真实缺陷，不是无害优化。

---

## 二、修复决策：统一到解码器一侧

两个方向：

- (A) 解码器改回精确激活 —— 但 CPU 解码会慢 **49.5×**（53.5 µs/token → 1.08 µs/token），
  且该近似本就是解码路径的既定性能选择。
- (B) **训练器改用同一 Padé 近似** —— 保住解码速度，同时恢复 R4。

**采用 (B)**（用户决策：该近似对 CPU 解码友好）。

### 改动

`dual_state_cuda_resident.cuh` 新增（与 `cpu_fast_activation.hpp` 逐位同式）：

```cpp
__device__ float ds_fast_tanh(float x){
    if(x> 3.0f)return  1.0f;
    if(x<-3.0f)return -1.0f;
    const float q=x*x;
    return x*(27.0f+q)/(27.0f+9.0f*q);
}
// f(x)=(27x+x³)/(27+9x²) 的解析导数
//   f'(x) = (729 - 162x² + 9x⁴) / (27 + 9x²)²
//   校验：f'(0)=1（与 d/dx tanh 一致）；f'(3)=0（与夹紧区平滑衔接）
__device__ float ds_fast_tanh_grad(float x){
    if(x>3.0f||x<-3.0f)return 0.0f;
    const float q=x*x,den=27.0f+9.0f*q;
    return (729.0f-162.0f*q+9.0f*q*q)/(den*den);
}
__device__ float ds_fast_sigmoid(float x){return 0.5f*(1.0f+ds_fast_tanh(x));}
__device__ float ds_fast_sigmoid_grad(float x){return 0.5f*ds_fast_tanh_grad(x);}
```

再包一层 `ds_act_tanh` / `ds_act_sigmoid` / `ds_act_tanh_grad` / `ds_act_sigmoid_grad`，
用 `TAO_TRAIN_FAST_ACT`（**默认开启**）切换；`-DTAO_TRAIN_EXACT_ACT` 可回退。

**关键约束：不改动 `ds_sigmoid` 本身。** `mem.beta` 与 `silu` 必须保持精确，
因为 CPU 解码器的 beta 用精确 `sigmoid`（`dual_state_cpu.hpp:44`），改了就再次引入不一致。

三处调用点同步替换：

| 文件 | 位置 | 改动 |
|---|---|---|
| `dual_state_cuda_resident.cuh` | `ds_update` 前向 | `ds_sigmoid/tanhf` → `ds_act_*` |
| `dual_state_cuda_backward.cuh` | `ds_update_backward` 反向 | 同步用解析导数 |
| `gpu_fixed_validation_arch2.cuh` | `state_update` 验证前向 | 同步（否则验证 NLL 与训练 NLL 系统性偏离） |

> 反向**必须**用同一函数的解析导数，否则前向/反向不一致，训练会**静默**劣化。

---

## 三、验证：`act_parity_test.cu`（永久守卫）

不靠"看起来对"，而是逐点比对。检验四件事：

| # | 检验 | 结果 |
|---|---|---|
| 1 | 前向一致：GPU `ds_act_*` vs CPU `fast_*`（400 001 点，含 ±3 夹紧边界） | max **1.192e-07** |
| 2 | 导数正确：`ds_act_tanh_grad` vs 双精度中心差分 | max **1.925e-07**，超差 0 |
| 3 | 门控更新一致：GPU `ds_update` vs CPU `fast_gated_update`（100 000 点） | max **7.749e-07** |
| 4 | **量化旧缺陷**：旧训练器（精确）vs 解码器（Padé） | tanh **2.352e-02** / sigmoid **1.592e-01** |

**1.192e-07 恰好是 1.0 处的 1 ULP**（2⁻²³），来源是 nvcc 的 FMA 收缩，
属跨 ISA 正常舍入，**不是语义差异**。

修复前后差距：

| 量 | 修复前 | 修复后 | 倍数 |
|---|---:|---:|---:|
| tanh 最大偏差 | 2.352e-02 | 1.192e-07 | **197 000×** |
| sigmoid 最大偏差 | 1.592e-01 | 1.192e-07 | **1 336 000×** |

判据因此设为**语义等价**（≤ 4 ULP），而非跨 ISA 逐位相等 —— 后者不可达且无意义。

```
PARITY_OK trainer and decoder use the same operator (rounding-level agreement only)
```

**该测试必须随每次架构改动重跑。** 构建：`scripts/build_act_parity.bat`。

---

## 四、算子版本规则（重要）

激活变了 ⇒ **算子变了** ⇒ 检查点不通用。

| 检查点 | 训练算子 | 解码时必须 |
|---|---|---|
| `h2r_l6_long/*`、`h2r_l6_run/*`、`ds3_l6_run/*`、`copy_order_run/*` | 精确 | **`TAO_FAST_ACT=0`** |
| 本次修复后训练的所有检查点 | Padé | 默认（`TAO_FAST_ACT=1`） |

解码器 `applyDefaultArch()` 保持 `fast_act` 默认 **true**（解码路径选择），
`TAO_FAST_ACT=0` 作为旧检查点的兼容开关。

> 历史评测（doc 21/22 的 held-out 数字）都是在**不一致**状态下测的。
> 那些结论的方向（泛化不差、顺序缺失）不受影响 —— 因为差异只有 2e-2 量级，
> 而缺陷表现是**完全失败**。但绝对数字需在新算子上重测。

---

## 五、R2 证明开关生效

训练器启动时打印：

```
ACTIVATION pade-fast  (decoder must use default TAO_FAST_ACT=1)
```

实测五次容量扫描运行全部打印该行，证明 `TAO_TRAIN_FAST_ACT` 确实生效
（满足 R2「证明开关生效」，不靠"应该开了"）。
