# DiscreteModel 交接给 MiniMax M3.1（2026-10-10）

## 0. 接手目标与当前结论

工作目录：`C:\Users\Administrator\Desktop\DiscreteModel`。用户希望继续分析并修复 Engram-Rule 模型的学习问题，重点是**只给问题时能否正确生成训练答案，以及对严格留出问答能否泛化**。本目录不是 Git 仓库；不要假设有可用的 `git diff`、分支或可回滚提交。历史 checkpoint 和 JSONL 均应保留，新的实验使用新文件名。

当前已经修复**评测和若干训练/生成接线错误**，但**没有修好留出问答质量**。关键区别：专门训练 8 条问答时，足额生成预算下 3 个种子都能完整生成 8/8 条训练答案；v11 同时训练 820 条问答时，全部训练答案的 teacher-forced token accuracy 为 79.05%，固定抽取 8 条训练题只给问题自由生成仅 1/8 完整正确。v11 的固定 12 条留出题仍为 0/12。不能把 8 题记忆能力、v11 训练集拟合、严格留出泛化混成同一个结论。

**先读顺序：**`AGENTS.md` → 本文 → `engram_rule/DESIGN.md` §30–§38 → `engram_rule/model.py` → `engram_rule/dialogue_small.py` → `engram_rule/memorize_dialogue_8.py` → 两个 `check_*.py`。`DESIGN.md` 的旧章节保留了被否证的假设，引用旧读数前必须看后续撤回。

## 1. 不可跳过的项目纪律

`AGENTS.md` 是上位规则。每轮先写设计、前提和**可证伪预测**；改前读代码，改后复读；**改代码 → 补可失败检查 → 检查通过 → 才准跑模型/对比**。每个诊断须有负向对照，证明反例出现时指标确实改变；常量指标不是证据。检查失败先读检查本身。所有 `while` 有硬上限和断言；全量检查 train/eval 的问题、答案及相关词项泄漏；报告多种子极差，差异小于极差不下结论。数值对拍按机器精度说明容差。大作业前清理残留进程、监控 RSS，训练单进程串行，默认有效 batch ≥256；性能结论须附 CUDA/CPU profile 读数。

上述规则不是让你机械重复旧实验：先核对**判据是否可失败、参考答案是否在生成预算内、任务是否真的有区分度**。不能因任务困难或上下文长而提前停止，也不能把假设写成已证实修复。

## 2. 当前实现究竟是什么

主模型见 `engram_rule/model.py:467-486`。输入 token IDs 经可训练 Embedding；Engram 对当前位置结尾的 1..n 元 token ID 做多哈希，候选槽用 `alive` 掩码筛选；可训练 token 嵌入之和作查询，对存活候选按 `-β‖query-key‖²` softmax，得到**逐候选**规则值、权重和 skip（`model.py:151-241`）。这次检索在 block 循环**外**，同一组规则结果供全部 block 使用。

每个 `RuleBlock`：`LN → RuleAttention → 残差 → LN → Linear-GELU-Linear FFN → 残差`（`model.py:447-464`）。RuleAttention 将候选值先逐个 sigmoid 门控、再按软读权重聚合为 `agg`；它分两条并联支路（`model.py:355-444`）：

1. `direct(agg)` 直接加到注意力输出，不经过窗口。
2. `bucket_proj(agg) + pos_bucket[绝对位置]` 得桶偏置，按桶标准化后 **argmax 只选一个起点桶**，因果夹紧并取连续 `window_max` 个回看位置；Q/K 部分维度做 RoPE，窗口内 K/V 经过带内容门、绝对位置门和桶强度的 SiLU 低秩门控，默认按 `Σ gate·up / (Σ |gate| + ε)` 逐通道局部归一化，再投影回模型宽度。

**不要按过时注释把它画成 top-4 桶或标准全序列 self-attention。** `k_buckets=4` 当前只被赋值，forward 用的是 `argmax` 一个起点（`model.py:299,388-400`）；序列位置间没有 `QKᵀ`/`L×L` softmax。Softmax 仅用于 Engram 候选槽。无存活候选时注意力与 direct 支路为零，FFN 仍运行。末端是 LayerNorm 和与输入 Embedding **不绑权**的无偏置词表线性头。

v11 问答 checkpoint（`engram_rule/engram_rule_dialogue_bpe_answer_disjoint_v11.pt`）的**实际配置**：词表 1024、宽度 64、2 层、4 头（每头 16）、RoPE 8 维、FFN 隐层 128；Engram 1–2 元 × 2 哈希＝每位置 4 候选、4096 槽、`min_count=1`；16 桶、桶步长 32、连续窗口 16、门控秩 16、位置表长度 512；总参数 767,278。通用 `RCfg` 默认是另一套更大配置，`train.py` 的 Zipf 合成基准也不是这份 v11 问答模型。问答配置入口见 `dialogue_small.py:1102-1105`。

## 3. 问答数据、监督与测量口径

`dialogue_small.py:106-115` 将记录排为 `[BOS][USER]问题[ASSISTANT]答案[EOS]`（可带历史，但 v11 抽样历史为 0）；ByteLevel BPE 仅用训练 split 拟合，训练和留出完整记录必须逐条往返无损。问题或答案规范化后共享任一项的记录合并为连通组件，再做全量**完整问题/完整答案字符串**隔离 split（`dialogue_small.py:835-894`）。这不保证拆分后的词、实体或子串跨 split 不相交；`AGENTS.md` 要求的词项级隔离仍需另行全量审计，不能把当前 split 误报成已满足该层条件。v11 的固定 revision/样本摘要见 `DESIGN.md` §36：抽样 1,024 行，train 820、eval 204，样本 SHA-256 `698e48bedcadaec30d32d20a2ffd6605557fec32bb7537fa8069e6681159a10c`，seq 512。

`answer_loss_sums` / `train_dialogue`（`dialogue_small.py:358-405`）只监督答案及 EOS 的下一 token CE，问题前缀不计 loss；v11 `first_token_weight=1`，AdamW + OneCycle、200 updates、有效 batch 256、lr 0.003、seed 11。teacher-forced 指标允许看到**参考答案已给的先前 token**，不能当成只给问题的回答正确率。greedy generation 从问题 prefix 开始逐 token 生成，必须同时检查 EOS、长度/上下文上限，以及完整记录减 prefix 的参考 token 数（`dialogue_small.py:550-688`）。

## 4. 已完成修复与可核验证据

| 项目 | 已完成状态与证据 |
|---|---|
| 8 题记忆评测预算 | 原报告只给 32 token，5/8 训练参考答案根本无法完成；`memorize_dialogue_8.py:31-36,103-104` 现于训练前验证答案含 EOS 能放进预算，独立 v3 实验用 192 token。相同旧权重、同一解码器只改 32→192，训练 EM **3/8→8/8**，超预算 **5/8→0/8**；`DESIGN.md` §37。 |
| 参考预算与上下文诊断 | `dialogue_small.py:550-688` 现按完整记录编码减 prefix 精确计参考 token；`truncated_context` 不再恒为 `False`；超上下文生成不再裁掉前缀重置绝对位置。`check_dialogue_small.py:150-200` 用“上下文先耗尽”与“主动 token 上限先耗尽”双向对照。旧报告中的零 `truncated_context_rate` 不能再作为证据。 |
| 训练目标接线 | `train_eval_dialogue` 已把 `first_token_weight` 传入真正的训练函数；`answer_loss_sums` 增加形状约束；相关负控在 `check_dialogue_small.py`。权重 8 只明显改善训练集首 token 拟合，留出提升没有超过种子极差，不能叫泛化修复；`DESIGN.md` §36。 |
| 数据和采样 | train-only BPE、严格输入文本审计、完整记录不左裁、问题/答案组件隔离、temperature>0 的每题独立 RNG 已修；同 prompt 多答案保留并报告，不擅自删样本；`DESIGN.md` §33–§35。 |
| 有界查找 | `split_records` 并查集的无界 `while` 改为最多 `len(rows)` 次并断言异常链；`dialogue_small.py:835-847`。 |

最后一次源代码改动后，系统 Python 3.14 / PyTorch 2.14.0+cu126 的 `py_compile`、`python engram_rule/check_dialogue_small.py`、`python engram_rule/check_memorize_dialogue_8.py` 均通过，**之后没有再改源代码**。`.venv` 未安装 PyTorch；根目录 `pyproject.toml` 只列 numpy，并未完整声明本问答脚本使用的 torch/tokenizers/requests。接手后若改源代码，必须重新按上面的先检查后实验顺序执行。

## 5. 当前可信读数：训练拟合和留出泛化分开看

| 实验/口径 | 训练集 | 留出集 | 边界 |
|---|---:|---:|---|
| v11，820/204，teacher-forced 答案 NLL | **1.19445** | **5.29376** | 单 seed 11；不能给种子极差。`dialogue_qa_answer_disjoint_v11.jsonl`。 |
| v11，teacher-forced 答案 token accuracy | **0.79046** | **0.25600** | 全量 820/204。 |
| v11，teacher-forced 首答案 token accuracy | **0.55244** | **0.10784** | 全量 820/204。 |
| v11，只给问题自由生成 | 固定抽取 8 条中 **1/8** 完整正确，F1 **0.34164** | 固定 12 条中 **0/12**，F1 **0.14885** | 训练抽样 8 条用 192-token cap，全数参考可完成、EOS 8/8、无 cap；报告 `dialogue_qa_answer_disjoint_v11_train_fit_review.jsonl`。历史留出 12 条用 140 cap，其中 1 条参考需 253 token，EM 在该行不可能成功；其余仍多离题。抽样 EM 不代表全部 820 题的生成正确率。 |
| 专门 8 题记忆、三 seed 101/202/303 | 每种子 **8/8**，EM 极差 0；训练 NLL 均值 **0.015415**、极差 **0.000238** | 每种子 **0/12**，EM 极差 0；留出 NLL 均值 **7.08193**、极差 **0.19228** | 同一 8/12 双隔离题、200 updates、bs 256、192-token cap；`dialogue_qa_memorize8_budget192_v3.jsonl`。 |

v11 训练指标已从保存的 1,024 原始行重建：820/204 划分和 checkpoint 中 204 条 eval 编码张量逐元素相同，全部 820 条 train teacher-forced 指标与旧报告差异 `<2e-6`。在固定 8 条训练题上把 8 个答案循环错配、保持答案集合不变，首 token accuracy **5/8→0/8**，首 token NLL **1.5934→7.8730**；这是可失败负控，证明这批训练题学到一定问题条件信号，但完整自由回答仍只有 1/8。详见 `DESIGN.md` §38 和上述 train-fit JSONL。v11 无多种子 checkpoint，不能从这 8 条推断稳定比例。

## 6. 已撤回的判断，不要重复引用

1. **撤回“teacher-forced 已完美拟合但 8 条训练题自由生成只有 2–3/8，所以存在剩余暴露偏差/架构不能记忆”。** 旧 32-token cap 使 5 条答案不可完成；足额预算后三 seed 均为 8/8。旧数字保留为历史，但不能再当学习失败证据（`DESIGN.md` §30、§31、§37）。固定 `bucket_step` 的独立 logits 对拍仍成立，不能因 EM 误判而一并抹去。
2. **撤回旧 `truncated_context_rate=0` 作为上下文充足证据。** 它曾写死为 false。v11 这 12 行另有独立证据：最大的 `prompt_tokens + generated_tokens=305 < seq512`，EOS 12/12、长度 cap 0/12（§36）。
3. 权重 8 的三 seed 对照：训练首 token acc 从 0.5890 提到 0.9806，但留出首 token acc 0.0972→0.1250，差值小于种子极差 0.0833，留出 EM 两组均 0/12；**不是已证实的泛化修复**（§36）。
4. UTF-8-safe 解码、温度采样顺序、加大有效 batch、固定权重下改桶步长等都未证明能解决 heldout 的答非所问。历史 CFG/一致任务的“泛化优势”还因泄漏、任务捷径和错误判据多次撤回；若重新碰合成泛化线，先读 `DESIGN.md` §16–§26。

## 7. 尚未解决的问题与有界的下一步

**已观察到的事实：**v11 训练集完整自由回答仍不稳定，且 train/eval NLL 差约 4.10；这同时存在训练拟合不充分和留出泛化差距。不能只叫“欠拟合”，也不能仅凭一个 seed 断言继续训练必无效。静态审计未发现对话前缀/答案边界、EOS、优化器累积、首 token 位置的确定性错位；也不能因此证明架构没有缺口。随机初始化、仅 820 条训练问答对任意未见事实可能缺少监督信号，这是**假设**而非已证明的唯一原因。

**建议按顺序接手：**

1. **先固定可信评测口径。** 从 v11 checkpoint 保存的原始行重建 split/tokenizer，除完整问答字符串隔离外，再全量审计词项/实体的跨 split 重合并明确评测所需的隔离定义；逐题核对参考答案+EOS 的实际预算，并把超预算题单列。不要静默改变 12 题集合后与旧 0/12 直接比。固定 12 题中“松鼠桂鱼”参考需 253 token、旧 cap 140；“你喜欢园艺吗？”原始样本有不同答案；“雪山飞狐”参考有“苗若兰是自己的女儿”的内部矛盾（v11 JSONL 第 6、7、12 行）。这些缺陷不能解释其余题的离题输出。人工事实审计应与模型指标分开。
2. **测问题条件是否真的迁移。** 可对全部 204 条 eval 构造匹配与错配问答的前 K 答案 token 平均 logP；预登记配对差分 `Δ=S(qᵢ,aᵢ)+S(qⱼ,aⱼ)-S(qᵢ,aⱼ)-S(qⱼ,aᵢ)`，并用“评分仅依赖答案”的 oracle 得 Δ=0、“只奖励正确配对”的 oracle 得正 Δ，先证明判据可失败，再与固定种子随机错配分布比较。报告 seed 极差、答案预算、按题型/近邻支持分层。**这项诊断尚未实现，不能把建议当已测结论。**
3. **再判断优化不足还是泛化缺口。** 在完全相同的 820/204 数据哈希、BPE、seq、有效 batch、解码预算上预注册至少 3 种子的训练步数曲线；先过检查，再串行训练。预测：若训练 EM 随步数提高而留出指标不超过种子极差，则主要是记忆加强而非泛化修复；若二者稳定提高，说明原训练预算不足；若训练 NLL 降低但自由生成不动，应继续查首 token/前缀续写。新评测需为每条参考预先证明可完成。
4. **只有确认可迁移监督存在后再选修法。** 可研究问答配对排序损失、可靠语料预训练或窗口覆盖消融，但这些目前都是待验方案，不能直接把 `first_token_weight=8`、窗口 16→64、检索/模板答案或解码惩罚写成“修好”。任何架构/目标变更都要保留旧 checkpoint、设新报告名、给多种子极差和负控。

另有一个**与对话 v11 无关**的具体待修点：`model.py:196-211` 的 `EngramMemory.build_from_corpus` 按 token 块统计时，在块边界重置 n-gram 上下文，可能漏掉跨块真实 n-gram 并引入零前缀伪 n-gram；仅 `train.py` 合成训练入口调用它，对话用 `build_engram_from_examples`。若处理此问题，先造“重复 n-gram 专放块边界”的正/负对照，不要误称它解释 v11 留出失败。

## 8. 入口、检查与产物

在仓库根目录使用系统 `python`。**以下是接手后的检查命令，不代表本交接又重跑了训练：**

```powershell
python -m py_compile engram_rule/model.py engram_rule/dialogue_small.py engram_rule/memorize_dialogue_8.py engram_rule/check_dialogue_small.py engram_rule/check_memorize_dialogue_8.py
python engram_rule/check_dialogue_small.py
python engram_rule/check_memorize_dialogue_8.py
```

| 文件 | 用途 |
|---|---|
| `AGENTS.md` | 项目纪律；优先级最高。 |
| `engram_rule/DESIGN.md` | 历史实验、被否证假设与 §36–§38 最新判决。 |
| `engram_rule/model.py` | Engram-Rule 网络、检索、规则窗口与 block。 |
| `engram_rule/dialogue_small.py` | 数据源、split、BPE、训练损失、评测与生成。 |
| `engram_rule/memorize_dialogue_8.py` | 8 题记忆对照、预算前置断言、checkpoint reload。 |
| `engram_rule/check_dialogue_small.py`, `engram_rule/check_memorize_dialogue_8.py` | 离线检查及负向对照；失败先审检查本身。 |
| `engram_rule/engram_rule_dialogue_bpe_answer_disjoint_v11.pt` | v11 seed11 模型；保存完整 1,024 原始行、tokenizer、eval 张量。 |
| `engram_rule/dialogue_qa_answer_disjoint_v11.jsonl` | v11 全量 teacher-forced 摘要及 12 条 heldout 生成。 |
| `engram_rule/dialogue_qa_answer_disjoint_v11_train_fit_review.jsonl` | v11 820 条 train 重算、8 条 train 自由生成及错配负控。 |
| `engram_rule/dialogue_qa_memorize8_budget192_v3.jsonl`, `engram_rule/engram_rule_dialogue_memorize8_budget192_v3.pt` | 8 题三 seed 足额预算结果与 seed101 权重。 |
| `engram_rule/dialogue_qa_memorize8_fixedstride_v2_cap32_recheck.jsonl`, `engram_rule/dialogue_qa_memorize8_fixedstride_v2_cap192_reload.jsonl` | 同权重预算负控。 |

**交接完成标准：**接手者能从上述文件区分“已修的评测错误”“v11 训练集还没完整拟合”“严格留出泛化未成立”，并在任何新实验前明确负向对照、预算可达性、数据隔离、种子极差和资源上限。不要覆盖旧证据，也不要把假设包装成修复结果。
