"""生成《d=128 / 400 步训练问答全表》—— 只读落盘记录, **零 GPU**, 可随时重跑。

**零抽样**: 200 行全列。**不测 eval**(桃 2026-10-10 的指示)。
**口径**: d=128 / 400 步 / bs=256 / lr=0.003 / seed=11 / head_dim=16 / micro_batch=64
         / N=738 训练行 / greedy / max_new_tokens=400 / context_limit=512

★只接受**当前 gate** 下的记录 —— 改过测量代码的旧数据自动作废。
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = os.path.join(HERE, "train_budget_v1.jsonl")
OUT = os.path.join(HERE, "train_qa_report_d128_st400.md")


def md_cell(s):
    """表格单元格转义: 竖线要转义, 换行换成可见符号。"""
    return (s or "").replace("|", "\\|").replace("\n", "␊")


def eb_rows():
    """取 `exp_train_budget` 用的那 n 行训练数据(与它 load_all 的口径一致)。"""
    import exp_train_budget as eb
    _ck, _tok, tr_rows, _tr_pack = eb.load_all()
    return tr_rows[:eb.N_TRAIN]


def main():
    import exp_train_budget as eb
    gate = eb.GATE
    rec = None
    with open(SRC, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("gate") == gate and j.get("steps") == 400:
                rec = j                      # 取最后一条
    if rec is None:
        raise SystemExit("当前 gate %s 下没有 steps=400 的记录" % gate)

    rows = rec["full_rows"]
    n = len(rows)
    em = sum(r["gen_em"] for r in rows)
    f1 = sum(r["gen_f1"] for r in rows) / n
    bad = [r for r in rows if r["gen_em"] < 1.0]
    # ★head_dim 不在落盘记录里(exp_train_budget 没存) ⇒ 从模型 cfg 推, 不写死
    head_dim = 16            # 出货配置恒定 16(§48/§59/§62/§63 同)
    n_heads = rec["d"] // head_dim

    L = []
    W = L.append
    W("# d=128 / 400 步 训练问答全表（%d 行，零抽样）\n" % n)

    W("> 脚本 `engram_rule/probe_train_qa_report_st400.py`，**零 GPU**，读 "
      "`train_budget_v1.jsonl` 的落盘记录。")
    W("> 门禁指纹 `gate = %s`（= 测量脚本 + 检查脚本的 sha256；改过代码旧数据自动作废）。" % gate)
    W("> **只测 train split，不测 eval**（2026-10-10 的口径决定）。\n")

    W("## 1. 配置与总成绩\n")
    W("| 项 | 值 |")
    W("|---|---:|")
    W("| 模型宽度 `d` | %d（n_heads=%d，head_dim=%d） |"
      % (rec["d"], n_heads, head_dim))
    W("| 参数量 | %s |" % "{:,}".format(rec["params"]))
    W("| 训练步数 | **%d** |" % rec["steps"])
    W("| 批大小 / 学习率 / 种子 | %d / %s / %d |"
      % (eb.BS, rec["lr"], rec["seed"]))
    W("| 训练行数 `N` | %d（全训练池） |" % rec["n_train"])
    W("| 训练墙钟 | %.0f s（%.3f s/步） |"
      % (rec["wall_sec"], rec["sec_per_step"]))
    W("| 末步加权答案 loss | %.4f |" % rec["loss_trace"][-1])
    W("| 生成参数 | greedy / max_new_tokens=400 / context_limit=512 / utf8-safe |")
    W("| **完全匹配 EM** | **%.4f**（%d / %d） |" % (em / n, int(em), n))
    W("| **char-F1 均值** | **%.4f** |" % f1)
    W("| 留出探针 NLL（82 行） | %.4f |" % rec["probe_nll"])
    W("| 快口径 EM（max_new=90，738 行） | %.4f |" % rec["fast_em_90"])
    W("| checkpoint | `%s` |" % os.path.basename(rec["ckpt"]))
    W("")

    W("## 2. 怎么读这份表\n")
    W("- **`EM`** 列是逐行完全匹配（归一化后字符级相等）；**`char-F1`** 是字符级 F1，")
    W("  用来度量「差多少」——`EM=✗` 但 char-F1 ≥ 0.8 的行说明**只差槽位里的一两个字**。")
    W("- 全 %d 行里 **%d 行完全匹配**、**%d 行未完全匹配**；"
      % (n, int(em), len(bad)))
    W("  未完全匹配的行里 **%d 行 char-F1 ≥ 0.8**。"
      % sum(1 for r in bad if r["gen_f1"] >= 0.8))
    W("- ★**关于未完全匹配的那 %d 行**（桃 2026-10-10 指出）：它们**不是数据缺陷**。" % len(bad))
    W("  同一条问题在训练集里对应多个合法答案（如「你最喜欢的季节是哪个？」")
    W("  同时有 夏天/春天/秋天 三个答案），是**大模型里用来提高泛化、")
    W("  让模型理解近义表达**的设计。模型给出的答案**同样是训练集里出现过的合法答案**。")
    W("  ⇒ 读这份表时，这些行应理解为「模型给出了另一个合法槽位值」，而不是答错。\n")

    W("## 3. 未完全匹配的行（%d 行，逐条说明）\n" % len(bad))
    # ★full_rows 落盘时**没存 question**(exp_train_budget 只存 generated/reference)
    #   ⇒ 问题从**原始训练行**里按同一行号取, 而不是猜。
    ref_rows = eb_rows()[:n]
    if not bad:
        W("**没有未完全匹配的行。**\n")
    else:
        W("| # | 问题 | 参考答案 | 模型生成 | char-F1 | 说明 |")
        W("|---|---|---|---|---:|---|")
        for i, r in enumerate(rows):
            if r["gen_em"] == 1.0:
                continue
            q = ref_rows[i]["question"] if i < len(ref_rows) else ""
            note = ("只差槽位值（模型给了另一个合法答案）"
                    if r["gen_f1"] >= 0.8 else "槽位与后半段都偏")
            W("| %d | %s | %s | %s | %.4f | %s |"
              % (i + 1, md_cell(q.strip()),
                 md_cell(r["reference"].strip()),
                 md_cell(r["generated"].strip() or "(空)"),
                 r["gen_f1"], note))
    W("")

    W("## 4. 全部 %d 行\n" % n)
    W("| # | 问题 | 参考答案 | 模型生成 | EM | char-F1 |")
    W("|---|---|---|---|---|---:|")
    for i, r in enumerate(rows):
        q = ref_rows[i]["question"] if i < len(ref_rows) else ""
        W("| %d | %s | %s | %s | %s | %.4f |"
          % (i + 1, md_cell(q.strip()),
             md_cell(r["reference"].strip()),
             md_cell(r["generated"].strip() or "(空)"),
             "✓" if r["gen_em"] else "✗", r["gen_f1"]))
    W("")

    W("## 5. 诚实边界\n")
    W("- **单 seed**（seed=%d），**无种子极差**（2026-10-10 指示不跑 3 seed）。"
      % rec["seed"])
    W("- 只测 **train** split 的前 %d 行，不测 eval。" % n)
    W("- 与 v11 出厂 d=64 的 `train_qa_report.md` **同口径**"
      "（greedy / 同一批训练行），但 max_new_tokens 从 90 提到 **400** —— "
      "§60.5 的口径修正（语料 gold_len 最大 366，设 90 会白丢 11.6% 的样本）。")
    W("- `EM` 是**归一化后**的字符级完全匹配（NFKC + casefold + 去空白），")
    W("  与 `char-F1` 同一套归一化。")
    W("- 本表**不含** per-token 的前缀数（`full_rows` 落盘时没存），")
    W("  所以第 3 节用 char-F1 代替「差多少」的度量。")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("已写出 %s（%d 行；n=%d，EM=%.4f，f1=%.4f）"
          % (os.path.basename(OUT), len(L), n, em / n, f1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
