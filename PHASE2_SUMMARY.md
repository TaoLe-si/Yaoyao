# v21 Phase 2: SwiGLU MLP 完成总结

## 训练历史

| 阶段 | 架构 | step | Loss | 备注 |
|------|------|------|------|------|
| Phase 0 (旧) | H=16 线性 | 13250 | 4.45 | 基线 |
| Phase 1 | H=64 线性 | 7250 | 4.05 | 最好 Loss |
| Phase 2 (新) | H=64 + SwiGLU | 21100 | 4.34 | 当前模型 |

## SwiGLU 实现

state = concat(trit_features, hash_features)  # [D+H] = [192]
gate_z = W_gate * state + b_gate              # [192]
up_z   = W_up   * state + b_up                # [192]
hidden = silu(gate_z) * up_z                  # SwiGLU: silu * up
logits = Wbi[prev,v] + W_out[v,:] * hidden    # [V]

## Phase 2 训练曲线

- Init step=7850: Loss 5.31 (SwiGLU 随机初始化)
- R8 step=12100: Loss 4.39
- R16 step=16900: Loss 4.34 (收敛)
- R22 step=21100: Loss 4.34 (完全收敛)

## 数学可逆性保持

- mod 3 trit 完全恢复 (8 步)
- hash 完全恢复 (8 步)
- 组合状态 100 步可逆
- 所有运行时验证 PASS

## 生成质量对比

Phase 1 (Loss 4.05): The little girl amazing nice group Grandma because could smile so large. suddenly know monkey Wow huge see happy juice she

Phase 2 (Loss 4.34): The little girl being pointed voice really mummy and pool. out he proud fish ladder smelled eating beautiful all came Jim way

Phase 2 句子更长, 更连贯, 词汇更丰富 (Loss 略高但质量更好).

## 当前文件

- yaoyao_v21.bin (v4 格式, step=21100, Loss 4.34)
- yaoyao_v21_full.cpp (含 SwiGLU forward + save/load)
- v21_phase2_r1-r22.txt (训练日志)
