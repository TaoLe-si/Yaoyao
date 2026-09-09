# 双状态语言模型：第4至8步已批准

本记录承接DUAL_STATE_ARCHITECTURE.md及DUAL_STATE_NORMALIZATION_DECISION.md，以用户逐步批准为准。无训练/实验运行。

4. 每层双状态读出后接Nf->Wup->SiLU->Wdown->残差前馈，无新增持久状态，无本层即时状态回写。E/D待定。
5. 嵌入、门/候选、状态读出、前馈、词表主要矩阵三值+逐行FP32尺度；偏置、RMS增益、激活、持久状态FP32。GPU原生CUDA QAT，CPU同有效模型，不擅自硬化门。
6. 每优化器步骤开始投影一次，累积期间冻结有效权重。逐行排序abs，选择最大(sum_topk abs)^2/k，alpha=sum/k，符号topk其余零。abs同值按列索引，评分平局较小k，全零alpha1。非有限报错。恒等STE，不对选择和尺度求导；尺度由主权重重估非独立学习。导出最新主权重投影，CPU不重新量化。GPU高效排序及FP32评分精确复现仍是实现任务。
7. 正式子词+byte fallback，训练分区拟合，控制标记转义。输入输出绑定E，logits=E*Nfinal(x)+FP32词表bias。共享主权重梯度累加，不减少全词表打分成本。逐token解码无历史KV，有限状态不保证任意长记忆。TURN_END停止本次助手生成保留状态，EOS结束序列，新会话清零；贪心基准及temperature/top-p选项。尚需细化TURN_END到新用户的token消费与EOS监督兼容协议，不能丢失边界token更新。
8. next-token目标对齐mask；用户输入更新状态，助手目标计损失。连续块TBPTT，跨块状态保持且detach；padding不更新，不同样本重置，每槽独立。步骤内有效权重固定，步骤间状态不重算，承认旧权重产生状态的近似。按整个步骤监督token总数归一化，无监督则推进状态不更新。步骤边界checkpoint含主权重/优化器/尺度规则/数据位置/RNG/各槽状态，不保存半步梯度。块长批量及优化器超参待硬件预算。

下一步待讨论：门偏置、状态与残差初始化。不是新增测试矩阵；一次定稿后接GPU/CPU实现。
