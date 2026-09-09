# 双状态模型第11至14步：批准记录

11. 本机RTX4070Laptop，8187MiB，compute8.9，CUDA13.0及MSVC14.44原生查询通过。首版L8 D512 S128 M512 E1024 V16384(含特殊token)。块256，微批4，累积8。主要矩阵约30146560权重，每层2719744；FP32状态20KiB/会话。非性能承诺。

12. AdamW beta .9/.999 eps1e-8 peakLR3e-4，矩阵主权重decay.01，偏置及RMS增益无decay。整个步骤按监督token总数归一化后全局norm裁剪1。有效权重步骤内固定；无监督不更新；非有限立即停止。预热有效优化步骤2%，余弦降至peak10%。总训练步数预定写配置，整数步预热边界须实现时明确。

13. FP32主权重正态初始化：暂存各投影sd1/sqrt(D+S)，长期各投影sd1/sqrt(D+S+M)，Ps/Pm sd1/sqrt(S+M)，up sd1/sqrt(D)，down sd1/sqrt(2LE)，绑定embedding sd1/sqrt(D)。down不重复缩放。词表bias0，候选bias0，暂存门bias0，长期门bias-2。RMS读出gamma1/sqrt(2L)，其他gamma1。保存初始主权重/RNG配置，首步前直接LS三值投影，不补偿投影方差。

14. 正式序列BOS USER...TURN_END ASSISTANT...TURN_END，多轮延续。生成TURN_END后恰好消费一次更新状态，不采用该次logits继续生成；保留状态至新USER。EOS只在有明确会话结束标注时监督；记录文件结束不是语义结束。此决策覆盖早期数据契约自动末尾EOS方案。已有pilot文件保留，正式适配器只移除其已知记录级末尾EOS，不任意删除内容或真实语义EOS；无此标注数据无需监督EOS。新会话重置再BOS，显式EOS生成后结束会话。长度限制/取消标截断，需继续则显式继续同消息，否则重建合法前缀。用户字面特殊字符串不变内部token。

当前：架构核心与训练算法决策已形成，尚非已实现系统。剩余明确工程规格包括tokenizer算法与训练产物、量化数值参考、模型/检查点容器及API，随后GPU训练到CPU解码集成。禁止改用CPU训练或重复符号优化。
