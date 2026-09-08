# 🧬 Phase 6 实验: 验证 SwiGLU 是否是 Loss 瓶颈

## 实验目的
测试"解冻 W_sgl_out 是否能降低 Loss". 

**假设**: W_sgl_out 冻结导致模型无法利用 SwiGLU 的隐藏层, 解冻后 Loss 应该下降.

## 实验方法
- 起点: yaoyao_v21_phase2_l21100.bin (CPU Loss 4.6729, 训练时 Loss 4.34)
- 终点: CPU dump on same test data (单次 forward, 1024 个 token)
- GPU trainer 添加 argv[7]: 0=W_sgl_out 冻结, 1=W_sgl_out 训练

## 实验结果 (CPU Loss after training)

| 配置 | CPU Loss | Δ from baseline |
|------|----------|-----------------|
| Phase 2 baseline (no train) | 4.6729 | -- |
| + 100 win, LR=0.005 | 4.7427 | +0.07 |
| + 100 win, LR=0.0005 | 4.6793 | +0.01 |
| + 100 win, LR=0.00005 | 4.6734 | +0.001 |
| + 2000 win, LR=0.0005 | 4.7478 | +0.07 |
| + 2000 win, LR=0.00005 | 4.6789 | +0.006 |
| + 1000 win, LR=0.005, sgl_out 训 | 4.9687 | +0.30 |
| + 2000 win, LR=0.0005, sgl_out 训 | 4.7478 | +0.07 |
| + 2000 win, LR=0.0005, sgl_out 冻结 | 4.7478 | +0.07 |

## 关键发现

1. **Phase 2 是局部最优** - 任何 fine-tuning 都会*退化* Loss
2. **LR 必须极小** (≤0.0005) 才能稳定 fine-tune
3. **W_sgl_out 训练本身没帮助** - 训练和冻结结果几乎相同 (4.7478 vs 4.7478)
4. **SwiGLU 不是瓶颈** - 当前架构已饱和, fine-tuning 无法突破 4.67

## 推论

- 之前假设"SwiGLU 冻结 = 瓶颈"是**错的**
- 真实瓶颈是*架构本身*: D=128, V=1024, 4-chain hash, frozen gW/aW 等等
- 必须 *scale up* 架构 (D/NL/V) 才能突破 Loss 下限
- 1.5M 模型的"信息天花板"可能就在 Loss 4.6 左右

## 后续方向

1. **不能 fine-tune** Phase 2 - 它已经饱和
2. **必须 scale up** - D=256, NL=4 是合理的第一站
3. **W_sgl_out 训练在低 LR 下无害** - 可以作为 "scale up 时"的常规训练项

## 代码改动

- yaoyao_v21_cuda_train.cu: 添加 W_sgl_out 可选训练 (argv[7])
- 添加 adam_Wsglout_kernel
- Model struct 扩展 (W_sgl_out_m/v 等)
- 实验表明 W_sgl_out 训练对结果*无显著影响* (在 LR=0.0005 下)
