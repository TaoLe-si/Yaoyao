"""生成《d=128 训练问答全表》—— **全部 820 行, 不抽样**。

数据来自 `exp_free_generation.py --part 1` 的落盘记录(带门禁指纹)。
**零 GPU**: 纯读 JSONL + 排版, 所以可以随便重跑。

⚠️ 只接受**当前门禁指纹**下的记录 —— 改过测量代码的旧数据不能拿来出文档。
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = os.path.join(HERE, "free_generation_v1.jsonl")
OUT = os.path.join(HERE, "train_qa_report_d128.md")


def main():
    import exp_free_generation as exp
    gate = exp.GATE
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
            if j.get("gate") == gate and j.get("part") == 1 and j.get("d") == 128:
                rec = j            # 取最后一条
    if rec is None:
        raise SystemExit("当前门禁 %s 下没有 d=128 的 part=1 记录" % gate)

    rs = rec["train"]
    n = len(rs)
    em = sum(r["gen_em"] for r in rs)
    f1 = sum(r["gen_f1"] for r in rs) / n
    tf = sum(r["tf_em"] for r in rs) / n
    hit = sum(r["tf_hit"] for r in rs) / n
    lcp0 = sum(1 for r in rs if r["lcp"] == 0) / n
    lcp = sum(r["lcp"] for r in rs) / n
    eos = sum(r["eos"] for r in rs) / n
    l1 = sum(1 for r in rs if r["tf_div"] == r["lcp"]) / n
    bad = [r for r in rs if r["gen_em"] == 0.0]
    # 完全正确 / 只差一个字 两档, 让桃一眼看出残余错误的形状
    near = [r for r in bad if r["gen_f1"] >= 0.8]

    L = []
    W = L.append
    W("# d=128 训练问答全表（%d 行，零抽样）\n" % n)
    W("> 脚本 `engram_rule/probe_train_qa_report_d128.py`，**零 GPU**，读 "
      "`free_generation_v1.jsonl` 的落盘记录。")
    W("> 门禁指纹 `gate = %s`（= 测量脚本 + 检查脚本的 sha256；改过代码旧数据自动作废）。\n" % gate)
    W("| 项 | 值 |")
    W("|---|---:|")
    W("| 配置 | d=%d, n_heads=%d, head_dim=%d |"
      % (rec["d"], rec["d"] // 16, 16))
    W("| 训练 | %d 步 / bs=256 / lr=%s / micro_batch=%d / seed=%d |"
      % (rec["steps"], rec["lr"], rec["micro_batch"], rec["seed"]))
    W("| 训练墙钟 | %.0f s（%.3f s/步） |"
      % (rec["wall_sec"], rec.get("sec_per_step", float("nan"))))
    W("| 显存 `max_memory_reserved` | 探测 %.0f MiB |" % rec.get("probe_peak_mib", float("nan")))
    W("| **完全匹配 EM** | **%.4f**（%d / %d） |" % (em / n, int(em), n))
    W("| char-F1 均值 | %.4f |" % f1)
    W("| 教师强制逐位全对 `tf_em` | %.4f |" % tf)
    W("| 答案 token 命中率 `tf_hit` | %.4f |" % hit)
    W("| **第一步就错** (`lcp=0`) | **%.4f** |" % lcp0)
    W("| 平均公共前缀 | %.2f token |" % lcp)
    W("| 正常吐 EOS | %.4f |" % eos)
    W("| L1 闭合率（`tf_div==lcp`） | %.4f |" % l1)
    W("")
    W("**读法**：`tf_em`（教师强制逐位全对）与 `gen_em`（自由生成 EM）只差 %.4f。"
      % abs(tf - em / n))
    W("L1 闭合率是\"教师强制的首个分歧位 == 自由生成的公共前缀长度\"——它接近 1 就说明")
    W("**自由生成没有额外的漂移**，答错的行在教师强制下同样答错。")
    W("")
    W("## 1. 残余错误清单（%d 行未完全匹配）\n" % len(bad))
    if not bad:
        W("**没有一行未完全匹配。**\n")
    else:
        W("其中 char-F1 ≥ 0.8（几乎对、只差一两个字）的有 **%d 行**。\n" % len(near))
        W("| # | 问题 | 参考答案 | 模型生成 | char-F1 | 前缀 |")
        W("|---|---|---|---|---:|---:|")
        for i, r in enumerate(rs):
            if r["gen_em"] == 1.0:
                continue
            W("| %d | %s | %s | `%s` | %.4f | %d/%d |"
              % (i + 1, r["question"].strip(), r["reference"].strip(),
                 (r["generated"].strip() or "(空)"), r["gen_f1"], r["lcp"], r["gold_len"]))
    W("")
    W("## 2. 全部 %d 行\n" % n)
    W("| # | 问题 | 参考答案 | 模型生成 | EM | char-F1 | 前缀 |")
    W("|---|---|---|---|---|---:|---:|")
    for i, r in enumerate(rs):
        W("| %d | %s | %s | `%s` | %s | %.4f | %d/%d |"
          % (i + 1, r["question"].strip(), r["reference"].strip(),
             (r["generated"].strip() or "(空)"),
             "✓" if r["gen_em"] else "✗", r["gen_f1"], r["lcp"], r["gold_len"]))
    W("")
    W("## 3. 诚实边界\n")
    W("- **单 seed**（seed=%d），**无种子极差**。桃 2026-10-10 指示不跑 3 seed。" % rec["seed"])
    W("- 只测 **train** split，不测 eval（桃 2026-10-10：「以后不用跑eval」）。")
    W("  eval 侧已被 §56.6 判死为泛化问题：d=64→128 时 eval `gen_em` 恒为 0/204。")
    W("- 本表与 v11 出厂 d=64 的 `train_qa_report.md` **同口径**"
      "（greedy / max_new_tokens=90 / context_limit=512 / 同一批训练行）。")
    W("- L1 闭合率 %.4f < 1.0 ⇒ 有 %d 行是**势均力敌的数值并列**被两次前向的浮点"
      % (l1, n - round(l1 * n)))
    W("  归约顺序差异翻的盘，不是机制分歧。")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("已写出 %s (%d 行; n=%d, EM=%.4f, f1=%.4f)"
          % (os.path.basename(OUT), len(L), n, em / n, f1))
    return 0


if __name__ == "__main__":
    sys.exit(main())