"""生成《训练集问答表现》文档 —— **只报我们出货的模型 v11**。

桃 2026-10-10 定: 只看 ours, 不拿实验变体(d=128/d=256)凑数。
⇒ 本脚本**零训练**, 直接加载 v11 出厂权重。补三件 DESIGN §38 之前没有的东西:
  1. **train 侧的无脑基线**(§49 只量了 eval) —— 没有 floor 就不能说"绑定能力多强"
  2. **自由生成的问答实例**(teacher-forced 指标看不出生成质量, §38 只记了 train 1/8)
  3. 答案开头按频次分层的完整表 + 频次依赖斜率 β(§46 有分层, 但没跟基线并排)

Δ **不重算**, 直接读 §48 的落盘记录(桃 2026-10-10: Δ 置换是最贵的一段, 且已有锚点)。

输出: `engram_rule/train_qa_report.md`
"""
import json
import os
import sys
from collections import Counter, defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
OUT = os.path.join(HERE, "train_qa_report.md")
SEQ = 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_SHOW = 12


def main():
    from tokenizers import Tokenizer
    import exp_capacity_scaling as exp
    import probe_capacity_scaling as judge
    from model import RCfg, EngramRuleLM
    from dialogue_small import (split_records, encode_examples, answer_metrics,
                                greedy_generate, char_f1)

    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr_rows, ev_rows, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                        seed=29, return_meta=True)
    tr_x, tr_s, tr_l = encode_examples(tr_rows, tok, SEQ)
    assert bool((encode_examples(ev_rows, tok, SEQ)[0] == ck["eval_tokens"]).all()), "eval 锚不一致"

    # ── 1. train 侧无脑基线（闭式, 只用 train 统计, 完全不看问题） ─────────
    tr_a = [tr_x[i, int(tr_s[i]):int(tr_l[i])].tolist() for i in range(tr_x.shape[0])]
    tr_types = Counter(t for a in tr_a for t in a)
    tr_first = Counter(a[0] for a in tr_a if a)
    n_tok = sum(len(a) for a in tr_a)
    top1 = tr_types.most_common(1)[0][0]
    b1 = sum(1 for a in tr_a for t in a if t == top1) / n_tok
    exp_uni = sum(c * (c / n_tok) for c in tr_types.values()) / n_tok
    bypos, bg = defaultdict(Counter), defaultdict(Counter)
    for a in tr_a:
        for j, t in enumerate(a):
            bypos[j][t] += 1
            if j:
                bg[a[j - 1]][t] += 1
    b2h = b3h = bnt = 0
    for a in tr_a:
        for j, t in enumerate(a):
            c = bypos.get(j)
            if c:
                b2h += int(c.most_common(1)[0][0] == t)
            pred = a[0] if j == 0 else (bg[a[j - 1]].most_common(1)[0][0]
                                        if bg.get(a[j - 1]) else top1)
            b3h += int(pred == t)
            bnt += 1
    b2, b3 = b2h / bnt, b3h / bnt
    top1_tok, top1_first = tok.id_to_token(top1), tok.id_to_token(
        tr_first.most_common(1)[0][0])
    b4_self = tr_first.most_common(1)[0][1] / len(tr_a)

    # ── 2. v11 模型（零训练, 直接加载出厂权重） ──────────────────────────
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    # Δ 从 §48 落盘记录读, 不重算
    stored = None
    with open(exp.OUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if j.get("arm") == "main" and j.get("d") == 64 and j.get("seed") == 11:
                stored = j
    assert stored is not None, "容量实验里找不到 d=64 seed=11 的落盘记录"

    # ★口径: sub 必须用 keep 子集, 与 loo 同源(规则 23)
    loo, tr_keep, _, _ = exp.loo_and_subset(tok, tr_rows, SEQ)
    sub = [{"question": tr_rows[i]["question"], "answer": tr_rows[i]["answer"],
            "history": []} for i in tr_keep]
    m = answer_metrics(model, tr_x, tr_s, tr_l, exp.EVAL_BS, DEV)
    _, hit = exp.score_rows(model, sub, tok, SEQ)
    assert hit.numel() == loo.numel() == len(tr_keep), (
        "行子集错位: hit=%d loo=%d keep=%d" % (hit.numel(), loo.numel(), len(tr_keep)))

    # ── 3. 自由生成 ─────────────────────────────────────────────────────
    # ★greedy_generate 返回的是 dict, 键是 `generated`(不是 str)。
    #   本脚本第一版把它整个 repr 打进表里, 还拿那个 repr 去算 EM/char-F1
    #   ⇒ 那一版文档的第 4 节**全部作废**。
    gens = []
    for r in tr_rows[:N_SHOW]:
        g = greedy_generate(model, tok, r, max_new_tokens=90, device=DEV)
        assert isinstance(g, dict) and "generated" in g, "greedy_generate 返回结构变了"
        p = g["generated"]
        gens.append((r, p, char_f1(p, r["answer"]), g))
    em = sum(int(p.strip() == r["answer"].strip()) for r, p, _, _ in gens)
    f1m = sum(f for _, _, f, _ in gens) / len(gens)
    n_eos = sum(int(g["eos"]) for _, _, _, g in gens)
    n_trunc = sum(int(g["truncated_context"]) for _, _, _, g in gens)
    rep = sum(g["repeated_token_rate"] for _, _, _, g in gens) / len(gens)

    # ── 4. 组文档 ───────────────────────────────────────────────────────
    L = []
    W = L.append
    W("# 训练集问答表现（v11 出厂模型）\n")
    W("> 生成脚本 `engram_rule/probe_train_qa_report.py`，2026-10-10。**零训练**，")
    W("> 直接加载 `engram_rule_dialogue_bpe_answer_disjoint_v11.pt`。")
    W("> 数据与 v11 同源：`raw_dialogue_rows` + seed 29 按连通组件切分，**train 820 条**。\n")

    W("## 1. train 侧无脑基线（只看 train 统计，**完全不看问题**）\n")
    W("没有 floor 就不能说\"绑定能力有多强\"。这些是**闭式**统计量（直接 argmax），")
    W("不是训练出来的模型 ⇒ 它们是 floor 的**下界**。\n")
    W("| 基线 | 口径 | acc |")
    W("|---|---|---:|")
    W("| B1 | 恒定输出最高频 token `%s` | %.4f |" % (top1_tok, b1))
    W("| — | 按 train 频率**随机**抽的期望 | %.4f |" % exp_uni)
    W("| B2 | 按答案内位置取最高频 token | %.4f |" % b2)
    W("| B3 | train 答案内 **bigram**（只看上一个 token） | **%.4f** |" % b3)
    W("\n首 token 常数基线（= train 自身众数，闭式上界）：**%.4f**（`%s`）\n" % (b4_self, top1_first))

    W("## 2. teacher-forced 指标（train 820 条）\n")
    W("| 指标 | 读数 | vs 无脑基线 |")
    W("|---|---:|---:|")
    W("| 答案 NLL | %.4f | — |" % m["answer_nll"])
    W("| token acc | **%.4f** | 比 B3(bigram) **%+.4f** |"
      % (m["answer_token_accuracy"], m["answer_token_accuracy"] - b3))
    W("| 首 token NLL | %.4f | — |" % m["first_answer_nll"])
    W("| 首 token acc | **%.4f** | 比 B4(常数基线) **%+.4f** |"
      % (m["first_answer_token_accuracy"], m["first_answer_token_accuracy"] - b4_self))
    W("| Δtrain（问题条件信号） | **%+.4f ± %.4f** | 恒等问题对照下应精确 =0 |"
      % (stored["delta_train"], stored["delta_train_se"]))
    W("\n参数 %s（其中 Engram 占 %.1f%%）。"
      % ("{:,}".format(params), stored["engram_pct"]))
    W("Δ 直接取自 §48 落盘记录 `capacity_scaling_v1.jsonl`（arm=main, d=64, seed=11），")
    W("**本脚本未重算** —— 同一量只保留一个来源。\n")

    W("## 3. 答案开头按「留一频次」分层\n")
    W("LOO = 除它自己外，训练里还有多少行用同一个答案开头。")
    W("分层子集 n=%d（`sub` 与 `loo` 同源，已断言）。\n" % len(tr_keep))
    W("| 分层 | n | 首 token acc |")
    W("|---|---:|---:|")
    for (lo, hi), nm in zip([(0, 0), (1, 1), (2, 3), (4, 10 ** 9)],
                            ["LOO=0（只见 1 次）", "LOO=1", "LOO 2–3", "LOO≥4"]):
        mk = (loo >= lo) & (loo <= hi)
        W("| %s | %d | %.4f |" % (nm, int(mk.sum()), float(hit[mk].mean())))
    beta = judge.slope([float(v) for v in hit], [int(v) for v in loo])
    W("\n频次依赖斜率 **β = slope(命中率 ~ log(1+LOO)) = %+.4f**" % beta)
    W("⇒ 命中率**随这个开头被见过几次而上升**：这是\"绑定存在，但低频学不会\"的量化形状。")
    W("§48 证明 β 会随模型加宽被压到 0 附近（d=64 %+.4f → d=256 −0.0012）。\n" % beta)

    W("## 4. 自由生成（greedy，只给问题不给答案）\n")
    W("前 %d 条训练样本：**完全匹配 %d**，char-F1 均值 **%.4f**；"
      % (N_SHOW, em, f1m))
    W("其中 %d/%d 正常吐 EOS，%d 条撞上生成预算上限，平均重复 token 率 %.3f。\n"
      % (n_eos, N_SHOW, n_trunc, rep))
    W("| # | 问题 | 参考答案 | 模型生成 | EOS |")
    W("|---|---|---|---|---|")
    for i, (r, p, _, g) in enumerate(gens):
        W("| %d | %s | %s | `%s` | %s |"
          % (i + 1, r["question"].strip()[:20], r["answer"].strip()[:24],
             (p.strip() or "(空)")[:34].replace("\n", "␊"),
             "✓" if g["eos"] else "✗"))
    W("")

    W("## 5. 怎么读这份表\n")
    W("- **teacher-forced 分数高 ≠ 会生成。** token acc %.4f、首 token acc %.4f，"
      % (m["answer_token_accuracy"], m["first_answer_token_accuracy"]))
    W("  但 greedy 生成完全匹配只有 %d/%d —— 这是本项目的核心落差（§38 已记录 train 1/8），不是新问题。" % (em, N_SHOW))
    W("- **「看问题」的净贡献**：全答案 %+.4f、首 token %+.4f（相对闭式基线）。"
      % (m["answer_token_accuracy"] - b3, m["first_answer_token_accuracy"] - b4_self))
    W("  Δtrain=%+.4f 是更直接的证据：恒等问题对照下它精确为 0。" % stored["delta_train"])
    W("- **低频开头是真弱项**：LOO=0 组首 token acc %.4f vs LOO≥4 组 %.4f，β=%+.4f。"
      % (float(hit[loo == 0].mean()), float(hit[loo >= 4].mean()), beta))
    W("  §48 已证明这是**容量受限**并被治好；但**eval 侧迁移率不随容量变**（§48）。\n")

    W("## 6. 诚实边界\n")
    W("- **单种子**（v11 = seed 11），**无种子极差**。")
    W("- 自由生成取的是**前 %d 条原始顺序**，不是随机抽样；换一批样本结论可能变。" % N_SHOW)
    W("- B1–B4 都是**闭式**统计量，不是训练出来的 n-gram/LSTM ⇒ 是 floor 的**下界**，")
    W("  「模型比不看问题高多少」这个说法因此**偏保守**。")
    W("- 本文只报 **train 侧**。eval 侧的天花板见 DESIGN §49（无脑 floor 0.0833 / 0.1830）。")
    W("- 本脚本**不测**扩窗/加 Engram 等结构改动的影响（规则 17）。")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("已写出 %s (%d 行)" % (OUT, len(L)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
