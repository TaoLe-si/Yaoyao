# 中文语义组合实验 v0（非 transformer）

手写字词语义 + 显式组合规则 + 反馈学习 + 最小 RSI 闭环。
设计、形式化表示与可证伪预测见 [DESIGN.md](DESIGN.md)。

## 目录

| 文件 | 作用 |
|---|---|
| `lexicon.py` | 手写词条：语义类（含属性向量）、动词论元框架 |
| `semantics.py` | 句义帧构造 / 规范化 / 对拍 |
| `parser.py` | 非 transformer 的显式图表分析器（分词 + 15 条组合规则） |
| `data.py` | 训练 / 留出句（正例 + 语义不成立的负例） |
| `learner.py` | 离散感知机打分器（特征计数表） |
| `train.py` | 训练并保存 `weights.pkl` |
| `evaluate.py` | held-out 评测 + 基线 + 三组负向对照 + 留一动词 |
| `large_eval.py` | 300 条冻结大 held-out（12 类构式，按类均衡） |
| `manual_heldout.py` / `manual_eval.py` | **独立人工 held-out**（77 条，手写 gold，不用验证器） |
| `open_vocab.py` | OOV 降级验证（strict vs 开放词表） |
| `grow_lexicon.py` + `corpus_zh.txt` | 语料候选抽取 → 教师反馈 → 入典 |
| `cost_curve.py` | 26 个新动词的"反馈成本"曲线（每条词需要多少标签） |
| `rsi.py` | 验证器 / 打乱标签 / 纯自训练 / 主动学习 四组对照 |
| `verifier.py` | 手写验证器（角色—属性约束，同时是 RSI 上限） |
| `checks.py` | 静态检查 C1–C10（分词往返 / 正例可解析 / 构式覆盖） |

## 运行

```bash
cd zh_sem
python checks.py      # 先过检查（C1–C11）
python train.py       # 反馈学习
python evaluate.py    # 主 held-out（82 条）+ 基线/对照/LOVO
python large_eval.py  # 300 条冻结大 held-out（12 类构式）
python manual_eval.py # 80 条独立人工 held-out
python cost_curve.py  # 每词反馈成本曲线
python grow_lexicon.py# 语料候选 → 反馈 → 入典
python open_vocab.py  # OOV 降级
python rsi.py         # A/B/C/D 对照
```

当前规模：**154 个词形 / 156 个义项**、12 类构式、94 条训练句、
82 条人工 held-out、300 条冻结大 held-out、80 条独立人工 held-out。

所有读数落在 `eval_results.json` / `rsi_results.json`，不要手改。

## 它是什么 / 不是什么

- 是：一个**离散、可逐条审阅**的语义组合原型。每个句义都能回溯到规则 + 词条。
- 不是：不是要证明它比 transformer 强；不是"全部汉字"的词典（种子 45 词条）；
  不处理时态、量化、指代、隐喻、语篇。验证器是手写规则，RSI 的上限就是它。
