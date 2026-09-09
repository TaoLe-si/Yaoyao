# 双状态架构批准补充

用户已批准第二步内容门控更新及第三步归一化。本记录补充DUAL_STATE_ARCHITECTURE.md中的过时“待确认”措辞；本次旧文件编辑被观察策略拒绝，未假定写入成功。

已批准归一化：Nx(x)=gamma_x*x/sqrt(mean(x*x)+epsilon)，所有候选及门读取Nx(x)；残差直通保留原始x。r=P_s s+P_m m，y=x+Nr(r)。最终词表头前另有RMSNorm。状态s/m不额外归一化，读出不回写状态。epsilon、精度和初始化待定。小读出可能放大，残差流不保证有界。

第四步提案尚未批准：双状态块后增加无持久状态前馈分支。z=Nf(y)，f=W_down SiLU(W_up z)，x_next=y+f。Nf独立RMSNorm，f不另行归一化。扩展维E待CPU/GPU预算确定，不默认4倍。约2DE MAC/token/layer，不增加跨token状态，但训练须保存或重算中间激活。不直接回写本层s/m，会影响后续层输入。量化策略后续统一确定。不宣称此分支已验证有效。
