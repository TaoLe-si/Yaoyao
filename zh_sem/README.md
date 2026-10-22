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
| `evaluate.py` | held-out 评测 + 两组负向对照 |
| `rsi.py` | 验证器驱动的自训练闭环 |
| `checks.py` | 静态检查（分词往返 / 正例可解析 / 对照有效性） |

## 运行

```bash
cd zh_sem
python checks.py      # 先过检查
python train.py       # 反馈学习
python evaluate.py    # 主读数 + random_class / position_only 对照
python rsi.py         # 3 轮 RSI
```

所有读数落在 `eval_results.json` / `rsi_results.json`，不要手改。

## 它是什么 / 不是什么

- 是：一个**离散、可逐条审阅**的语义组合原型。每个句义都能回溯到规则 + 词条。
- 不是：不是要证明它比 transformer 强；不是"全部汉字"的词典（种子 45 词条）；
  不处理时态、量化、指代、隐喻、语篇。验证器是手写规则，RSI 的上限就是它。
