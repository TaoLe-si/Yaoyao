# 桃注意力 / Tao Attention

用户正式命名：桃注意力。现有实现为节点内无损保存最近K个token、准确逆恢复trit/Hash状态、三值历史混合。不是Transformer QK/softmax注意力，不保存KV。CPU推理通过节点副本逐步逆推；GPU训练可物化相同前缀状态以批量计算。

公式：m[t,d]=sum(k=0..min(K,t+1)) a[k,d]*r[t-k,d]，a属于{-1,0,+1}，初始前缀r[-1]=0。当前K16,D256,V1024；Hash特征仍只用当前Hash。CPU混合用加/减/跳过，预测头仍为浮点SwiGLU。

训练状态：独立坐标训练得到reader_stage_256.reader；该小批次读取器存在过拟合。完整GPU头适配分支yaoyao_reader_full_train.cu现训练Wg/Wu/Wo/Wbi，Q1和桃注意力系数本阶段冻结，不能说所有系数已联合训练。500次更新后 reader_full_step21502.bin，slice0的1024目标CE4.603831756->4.431794322。改善归因仍需等预算identity对照。

后续原则：优先连续训练和独立评估趋势；随机批次Loss波动不等于发散。门控仅在诊断需要时加入，先考虑三值门、固定2次幂缩放，避免破坏精确状态逆运算。模型规模和与Transformer效果必须报告真实参数量、公平数据/任务/预算/CPU时延基准，不预先宣称媲美。当前不是已验证的大规模模型。
