# 夭夭 (Yaoyao) v0.9 - 三值小语言模型

> 三值 {-1, 0, +1} 表示 + Q1 hash bucket + CPU 友好的小型语言模型

## 架构总览 (Q1-Q4)

```
输入 token --+-> Q1 hash bucket pool [B, K, D]  -+
             |                                    +-> x[t]  -> Q3 (k=3 conv) -> y[t]
             +---------- prev x[t-1] (query) ----+                    |
                                                                        v
                                                             Dynamic a = s(z/T), T=2
                                                                        |
                                                         +--------------+--------------+
                                                         v                             v
                                               h_new = a*h + (1-a)*y    s_new = s + y
                                                         |                             |
                                                         +------------ RMSNorm ----------+
                                                                        |
                                                                        v
                                  Q4: logits[v] = W_h*h + W_s*s + W_bi[prev,v]
                                                                        |
                                                                        v
                                                             LogSoftmax -> NLL loss
```

### 各组件作用

| 组件 | 数学公式 | CPU 友好性 |
|------|----------|-----------|
| **Q1 hash bucket** | softmax(q . trits[h*K+k]) 加权 K 个候选 | int8 存储, B*K*D*1 字节 |
| **Q3 (k=3 conv)** | y[t] = q3w0*x[t-2] + q3w1*x[t-1] + q3w2*x[t] | 局部上下文, 无 attention |
| **Q2-A h 通道** | h[t+1] = a*h[t] + (1-a)*y[t] (衰减记忆) | 标量门控, 极简 |
| **Q2-A s 通道** | s[t+1] = s[t] + y[t] (累积求和) | 标量累加, 极简 |
| **Dynamic a** | a = s(z/T), T=2.0, z = W_a*x + b_a | 每维独立 sigmoid |
| **Q4 输出** | logits[v] = W_h[v]*h + W_s[v]*s + W_bi[prev,v] | 矩阵乘法 |

### 关键设计原则

- **三值 (-1, 0, +1)**: 权重存储为 int8, 推理时可 SIMD 加速
- **无 attention**: 用 channel 状态 (h 衰减 + s 累积) 代替 attention
- **CPU 单核友好**: 单线程 forward, ~150 万 tokens/秒
- **小参数量**: D=64 NL=2 时约 15MB (含 Adam 状态)

## 文件结构

```
yaoyao_v09_word.cpp       - 完整训练 pipeline (word-level vocab + tinystories)
yaoyao_q1_step1.cpp       - Step 1: Q1 hash bucket 单测
yaoyao_q2a_step2.cpp      - Step 2: Q2-A channels + RMSNorm BPTT 单测 (SEQ=64)
yaoyao_q1_realquery.cpp   - Step 2b: Q1 backward 全梯度公式验证
yaoyao_v08_q1real.cpp     - 教学版: 字符级 + Q1 (deprecated)
docs/yaoyao_v06_architecture.md - v0.6 数学原理详细文档
```

## 编译

```bash
clang++ -O2 -std=c++17 -mavx2 -mfma -fopenmp \
    -o yaoyao_v09_word.exe yaoyao_v09_word.cpp
```

## 训练 (增量训练)

```bash
# 首次训练
./yaoyao_v09_word.exe

# 加载已有模型继续训练 (增量)
./yaoyao_v09_word.exe <text_path> <model_bin> [N_WIN] [EPOCHS]
./yaoyao_v09_word.exe tinystories_train.txt yaoyao_v09_model.bin 10000 3
```

模型二进制格式:
- 魔数 0x59414F59
- 版本 1
- V 维度
- 所有参数 + Adam 状态 (m, v) + step

## 单测结果

| 步骤 | 内容 | 通过 |
|------|------|------|
| Step 1 | Q1 forward + backward + chain rule | 16/16 |
| Step 2 | Q2-A channels BPTT (SEQ=64) | 49/49 |
| Step 6b | Q1 full gradient (direct + indirect) | 22/23 |

## 训练数据

- **TinyStories** (~500MB): 简单英文儿童故事, vocab=1024 word-level
- 词汇表从 sample 自动构建 (最常见的 1024 词)

## 当前最佳结果 (D=64, NL=2, V=1024)

| 指标 | 值 |
|------|-----|
| 参数量 | ~14.85 MB |
| 训练速度 | ~30 windows/sec |
| Loss (5 epochs) | 4.05 -> 3.67 |
| 显存 | 0 (纯 CPU) |

## 下一步计划

- [ ] D=128 NL=4 增大模型容量
- [ ] 字符级 + 词级混合 vocab
- [ ] 实际 BPE 子词分词
- [ ] Sequence packing (SEQ=128)
- [ ] 多任务学习: 续写 + 问答

## 许可证

MIT
