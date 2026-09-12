# 框架变更记录

## DONE
- ✅ ① 推理模式契约确立：`A 思考 <推理>答案：<X>`；`data/p1_reason.txt` 202,652 篇
- ✅ ② 特殊 token 扩展：THINK=261 / THINK_END=262 / FIRST_MERGE=263；8 个源文件同步；`build_corpus` 编译通过
- ✅ ③ 分词器 v2 训练中（16121 merges）

# 框架剩余缺口（按依赖顺序，不可跳步）

本文件是把 pipeline/config.json 中 status 非 done 的项展开成可执行清单。
上游未决 -> 下游重做，因此顺序不可颠倒。

## 1. 【已解】推理模式分隔符 -> 分词器 v2（原方案：控制 token，现改用文本标记）
现状：`src/language_data_contract.hpp` 只有 `BOS=256 USER=257 ASSISTANT=258 TURN_END=259 EOS=260`。
后果：模型没有任何机制进入/退出推理模式，只能把 CoT 当普通答案吐出 —— 这是评测格式率 0.00% 的根因之一。
改动面（这是它阻塞全部下游的原因）：
  - `language_data_contract.hpp` 加 THINK / THINK_END，Special 枚举后移
  - `Config::validate` 的 `vocab<261` 下界同步
  - 分词器训练脚本重建 -> 新 digest
  - 全部语料按新格式重建 -> 新分片
  - `train_shards` / `train_grpo` / `h2r_cpu` / `grpo_rollout` 全部重编
格式提案：`U <题> A 思考 <逐步推理> 答案：X`

## 2. ④ SFT 冷启动阶段（R1 第 1 阶段，完全缺失）
为什么必须补：RL 不能让模型从零学会一种格式。R1 靠冷启动 SFT 教会两段式，再让 GRPO 在格式内搜索。
好消息：`src/train_grpo.cu` 做的正是"对指定 span 做加权 CE"，这本身就是 SFT 训练器，
        只需换一个数据构造入口（按 思考/答案 切 span），不必新写训练器。

## 3. 【已解】⑤ GRPO 与论文对齐（三处已全部实现）
论文 Eq.3:  J = E[ 1/G * sum_i 1/|o_i| * sum_t min( ratio*A, clip(ratio,1-e,1+e)*A ) - beta*KL(pi||pi_ref) ]
  ✅ (a) 重要性比值 ratio = pi_theta/pi_theta_old 与 clip(1±e)
         -> 内核 ds_grpo_batch：r = exp(lp_theta - lp_old)，对 min(rA,clip(r)A) 求导
         -> 裁剪梯度经有限差分验证；A<0 的失效侧与 A>0 相反（早期版本写错过）
  ✅ (b) KL(pi_theta||pi_ref) 惩罚项 beta
         -> k3 估计，梯度项 beta*(e^d - 1)，d = lp_ref - lp_theta
         -> pi_ref 取冻结的 R1 冷启动 SFT 模型（驱动器传 --ref），各轮之间不移动
         -> 不可取 pi_old：那样 d≡0、KL 梯度恒为 0，会静默失效
  ✅ (c) 按序列长度归一化 1/|o_i|：取 w_i = A_i/|o_i|，配合 update() 的 1/supervised 全局缩放

实现要点：因 d(lp)/d(logits_j) = softmax_j - onehot_j 恰为 CE 的梯度，整个 Eq.3 归约为
逐 token 乘数 c_t = f'(r)*r + beta*(e^d - 1)，反向通路完全不必改。

仍与论文一般形式有偏差（如实记录）：内层更新次数取 mu=1，而非论文允许的 mu>1。

## 4. ③ 预训练（已就绪，等 1 完成后重建语料再跑）
配置：L2 d1024 s512 m1024 dk128 = 23,618,562 参数，595 位置/秒，批次 slots=16 width=16。
注意：`dv<=1024` 守卫使 d=m<=1024，这是 L2 容量天花板；若要继续加宽必须先改该守卫。
