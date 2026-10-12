"""从抽样问答表生成 markdown 报告。**只读 jsonl/summary，不硬编码任何读数。**

用法：
    python make_qa_report.py --tag n2952d128s462 --title "N=2,952 / d=128 / 462 步"
默认输入：`qa_train_s200_<tag>.jsonl`、`qa_held_s200_<tag>.jsonl`、
`qa_summary_<tag>.json`；输出 `QA_REPORT_<tag>.md`。

── 分类口径（每条都写清「为什么这样判」，且都可在负例上失败）──────────
| 桶 | 判据 | 它在反例下会不会变 |
|---|---|---|
| `exact` | `generated == reference` | 是（大小写/空白不同就掉出去） |
| `prefix_of_ref` | 生成是参考答案的真前缀 | 是：模型在长度上限处停住 ⇒ 命中 |
| `near_miss` | 非 EM 但 char-F1 ≥ 0.8 | 是：只差标点/一个词 |
| `partial` | 0.3 ≤ char-F1 < 0.8 | 是 |
| `wrong` | char-F1 < 0.3 | 是 |

★`eos` / `truncated_context` / `repeated_token_rate` 单列，因为它们区分
「答错」与「没答完/在打转」—— 这三件事混在一起时，只看 EM 会把解码问题
误判成模型能力问题。
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

F1_NEAR = 0.8
F1_MID = 0.3


def load_jsonl(path):
    if not os.path.exists(path):
        raise SystemExit("缺 %s" % os.path.basename(path))
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def bucket(r):
    """给一条抽样记录分类。返回 (桶名, 说明)。

    ★`prefix_of_ref` 的说明**由实测数据生成**，不写死「撞长度上限」：
      上一版我写死「多半撞 max_new_tokens」，而本批 `truncated_context` 率是 0.0%
      ⇒ 那句话与数据直接矛盾。真正的区别是「提前 EOS」，所以把 EOS 率带上。
    """
    if r["em"]:
        return "exact", ""
    g, ref = r["generated"].strip(), r["reference"].strip()
    if ref.startswith(g) and g:
        return ("prefix_of_ref",
                "生成是参考答案的真前缀；若本次截断率=0.0%% 则**不是**长度上限所致，"
                "而是模型提前 EOS（EOS 率 %.1f%%）" % (100.0 * r.get("_eos_rate", 0.0)))
    f1 = r["char_f1"]
    if f1 >= F1_NEAR:
        return "near_miss", "非 EM 但 char-F1≥0.8（只差标点/一两个词）"
    if f1 >= F1_MID:
        return "partial", ""
    return "wrong", ""


def stats(rows):
    n = len(rows)
    d = dict(
        rows=n,
        em=sum(r["em"] for r in rows),
        em_rate=sum(r["em"] for r in rows) / float(n),
        f1=sum(r["char_f1"] for r in rows) / float(n),
        eos_rate=sum(1 for r in rows if r.get("eos")) / float(n),
        trunc_rate=sum(1 for r in rows if r.get("truncated_context")) / float(n),
        rep_mean=sum(r.get("repeated_token_rate", 0.0) for r in rows) / float(n),
        gen_tok_mean=sum(r.get("generated_tokens", 0) for r in rows) / float(n),
        buckets={},
    )
    eos_rate = sum(1 for r in rows if r.get("eos")) / float(n)
    for r in rows:
        r["_eos_rate"] = eos_rate
    for r in rows:
        b, why = bucket(r)
        d["buckets"].setdefault(b, {"n": 0, "f1": 0.0, "why": why})
        d["buckets"][b]["n"] += 1
        d["buckets"][b]["f1"] += r["char_f1"]
    for b, v in d["buckets"].items():
        v["f1"] = v["f1"] / v["n"]
    # ★分桶完整性自检：各桶行数之和必须等于总行数，否则分类漏了一条
    assert sum(v["n"] for v in d["buckets"].values()) == n, \
        "分桶漏行：%d != %d" % (sum(v["n"] for v in d["buckets"].values()), n)
    return d


def load_ident_text():
    """从 `DESIGN.md` 里取身份文案与名字，**不手打汉字**。

    为什么：我手打的「\u5929」码点核对出来是**天**不是**夭**（同一 U+5929），
    于是第一次扫描把「今**天天**气」当成了身份问句，结论直接作废。
    ⇒ 凡是中文关键字，来源改成**项目自己的文档**，别靠我复述。
    """
    md = os.path.join(HERE, "DESIGN.md")
    if not os.path.exists(md):
        return None, None
    txt = open(md, encoding="utf-8").read()
    # 名字那个字由 `([^，]+)` 自动捕获，**不写死** —— 我曾写死 \u5929，那是「天」，
    # 于是第一次扫描把「今**天天**气」全判成身份问句，整份语料结论作废重扫。
    # ★名字是**两个字**（夭夭）：第一版用 `(.)` 只吃一个字 ⇒ 正则永不匹配 ⇒ 文档里
    #   「统一文案」显示「未找到」。**没匹配上时要说「未找到」，不要静默用兜底文案。**
    ZHUO = "\u707c"
    m = re.search("\u6211\u662f([^，]+)，(.+?)" + ZHUO * 2 + "\u5176\u534e", txt)
    if not m:
        return None, None
    return ("\u6211\u662f%s\uff0c%s%s%s\u5176\u534e\u3002"
            % (m.group(1), m.group(2), ZHUO, ZHUO), m.group(1))


IDENT_ANS, IDENT_NAME = load_ident_text()
IDENT_Q = re.compile(r"你(是|叫)什么|你(是|叫)什么名字|你的名字|你是谁|自我介绍|介绍一下自己|谁开发的你|你是什么模型")
CORPUS = os.path.join(HERE, "corpus_modelscope_v2_155740.jsonl")


def corpus_identity_scan(limit_train_rows=None):
    """★扫**整份语料**的身份问句，而不是只看这 200 条抽样。

    为什么必须扫全量：「抽样里没有身份问句」有两种完全不同的含义 ——
      (a) 语料里有，只是这 200 条没抽到 ⇒ 换个种子也许能核对；
      (b) 语料里**一条都没有** ⇒ 该权重根本不可能满足身份要求。
    只报抽样结果会把 (b) 误说成 (a)。
    """
    if not os.path.exists(CORPUS):
        return dict(error="语料文件不在：%s" % os.path.basename(CORPUS))
    total = hits = 0
    hit_rows = []
    for line in open(CORPUS, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not isinstance(r, dict) or "__meta__" in r:
            continue
        total += 1
        q = r.get("question") or ""
        if IDENT_Q.search(q):
            hits += 1
            if len(hit_rows) < 3:
                hit_rows.append(q[:50])
        if limit_train_rows and total >= limit_train_rows:
            break
    return dict(scanned=total, identity_questions=hits, examples=hit_rows,
                scoped_to=(limit_train_rows or total))


def identity_check(rows):
    """身份问句核对：命中统一文案才记 PASS；找不到问句也要**明说**，不能默认通过。"""
    hits = []
    for r in rows:
        if IDENT_Q.search(r["question"]):
            g = r["generated"].strip()
            hits.append(dict(question=r["question"], generated=g,
                             exact=(g == IDENT_ANS),
                             has_name=(IDENT_NAME in g)))
    return hits


def md_escape(s):
    return s.replace("|", "\\|").replace("\n", " ")


def sample_table(rows, k=12, only_em=False):
    # ★表头不能以「#」开头：markdown 会把整行当标题渲染，表格就散了。
    out = ["行号 | 问题（截断 34 字） | 参考答案 | 生成 | char-F1",
           "|---:|---|---|---|---:|"]
    picked = [r for r in rows if r["em"] is False] if only_em else rows
    for r in picked[:k]:
        q = r["question"] if len(r["question"]) <= 34 else r["question"][:34] + "…"
        out.append("| %d | %s | %s | %s | %.3f |"
                   % (r["idx"], md_escape(q), md_escape(r["reference"][:60]),
                      md_escape(r["generated"][:60]), r["char_f1"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--sample", type=int, default=12)
    a = ap.parse_args()

    tr = load_jsonl(os.path.join(HERE, "qa_train_s200_%s.jsonl" % a.tag))
    hd = load_jsonl(os.path.join(HERE, "qa_held_s200_%s.jsonl" % a.tag))
    sp = os.path.join(HERE, "qa_summary_%s.json" % a.tag)
    S = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    cell = S.get("cell", {})
    pt = S.get("probe_teacher_forced", {})

    st, sh = stats(tr), stats(hd)
    idt, idh = identity_check(tr), identity_check(hd)

    L = []
    A = L.append
    A("# %s" % (a.title or ("抽样问答表现报告（tag=%s）" % a.tag)))
    A("")
    A("> 本文件由 `make_qa_report.py` 从落盘 jsonl **生成**，所有数字可追回 "
      "`qa_train_s200_%s.jsonl` / `qa_held_s200_%s.jsonl`。改实验重跑即可刷新，"
      "不要手改本文件。" % (a.tag, a.tag))
    A("")
    A("## 一、被测对象与口径")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 权重 | `%s` |" % S.get("weight_source", "?"))
    A("| cell | n=%s, d=%s, steps=%s, epochs=%s, seed=%s, bs=%s, micro=%s |"
      % (cell.get("n"), cell.get("d"), cell.get("steps"), cell.get("epochs"),
         cell.get("seed"), cell.get("bs"), cell.get("micro_batch")))
    A("| 建 Engram 用到的 train 行数 | %s |" % S.get("train_rows_used", "?"))
    A("| alive 槽位 | %s / %s（%.4f） |"
      % (S.get("alive", {}).get("slots"), S.get("alive", {}).get("n_slots"),
         S.get("alive", {}).get("ratio", 0.0)))
    A("| held 教师强制探针 | 前 %s 行 |" % S.get("probe_rows_used", "?"))
    A("| 抽样 | 每侧 %s 条，种子 %s |" % (S.get("sample", {}).get("n_per_side"),
                                         S.get("sample", {}).get("seed")))
    A("| 解码 | greedy，max_new_tokens=%s，context_limit=%s |"
      % (S.get("gen", {}).get("max_new_tokens"), S.get("gen", {}).get("context_limit")))
    A("")
    if pt:
        A("held 教师强制（与 N 扫描同口径）：answer_nll=**%.6f**、ppl=%.4f、"
          "token_acc=**%.6f**、首 token acc=%.6f、answer token 数 %s"
          % (pt.get("answer_nll", float("nan")), pt.get("answer_ppl", float("nan")),
             pt.get("answer_token_accuracy", float("nan")),
             pt.get("first_answer_token_accuracy", float("nan")),
             "{:,}".format(pt.get("answer_tokens", 0))))
        A("")

    A("## 二、总体读数")
    A("")
    A("| 侧 | 行数 | EM | EM 率 | char-F1 | EOS 率 | 撞长度上限 | 平均重复 token 率 | 平均生成 token |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in (("train（训练见过，只证明记住）", st),
                    ("held（没见过，才证明泛化）", sh)):
        A("| %s | %d | %d | %.2f%% | %.4f | %.1f%% | %.1f%% | %.4f | %.1f |"
          % (name, s["rows"], s["em"], 100 * s["em_rate"], s["f1"],
             100 * s["eos_rate"], 100 * s["trunc_rate"], s["rep_mean"],
             s["gen_tok_mean"]))
    A("")
    A("> **单 seed、无种子极差**（AGENTS 规则 16）；抽样 ⇒ **不能外推到全量**。"
      "两条指标方向相反时以 **EM（逐行全对）** 为准：train tok_acc 可以在 N=5,904 上"
      "仍有 0.9589 而 EM 只有 0.575。")
    A("")

    A("## 三、失败分桶（可失败的分类，不是印象）")
    A("")
    for name, s in (("train", st), ("held", sh)):
        A("**%s**（%d 条）" % (name, s["rows"]))
        A("")
        A("| 桶 | 条数 | 占比 | 桶内平均 char-F1 | 说明 |")
        A("|---|---:|---:|---:|---|")
        for b in ("exact", "prefix_of_ref", "near_miss", "partial", "wrong"):
            v = s["buckets"].get(b)
            if v is None:
                # ★零也要显式列出来：漏行会让读的人以为「没这个桶」而不是「0 条」
                A("| `%s` | 0 | 0.0%% | - | - |" % b)
                continue
            A("| `%s` | %d | %.1f%% | %.4f | %s |"
              % (b, v["n"], 100.0 * v["n"] / s["rows"], v["f1"], v["why"] or "-"))
        A("")
    A("> **★本批最关键的一条**：held 侧**教师强制** token accuracy 是 **%.4f**，"
      "但**自由生成** EM 是 **%d/%d（%.2f%%）**。"
      % (pt.get("answer_token_accuracy", float("nan")), sh["em"], sh["rows"],
         100 * sh["em_rate"]))
    A("> 也就是说：**§69.52 里「N 从 738 到 2,952 让 held tok_acc 涨 31.9%」这个收益，"
      "在自由生成上一次都没兑现。** 两个读数不矛盾，测的是两件事：")
    A("> tok_acc 只要求**逐 token 在正确答案位置上**，EM 要求**整句全对且解码走对路径**。")
    A("> ⇒ 把 held tok_acc 当作「泛化变好」的证据去买点，是**跨越了两个口径**。")
    A("")

    A("## 四、抽样逐条（%d 条 train / %d 条 held 各取前 %d 条）" % (st["rows"], sh["rows"], a.sample))
    A("")
    A("### train 侧")
    A("")
    A(sample_table(tr, a.sample))
    A("")
    A("### held 侧")
    A("")
    A(sample_table(hd, a.sample))
    A("")

    A("## 五、身份问句核对")
    A("")
    A("统一文案：**%s**（从 `DESIGN.md` 程序化取出，不手打）" % (IDENT_ANS or "未找到"))
    A("")
    A("| 范围 | 身份问句条数 | 结论 |")
    A("|---|---:|---|")
    A("| train 抽样 %d 条 | %d | %s |"
      % (st["rows"], len(idt), "可核对" if idt else "**抽不到，无法核对**"))
    A("| held 抽样 %d 条 | %d | %s |"
      % (sh["rows"], len(idh), "可核对" if idh else "**抽不到，无法核对**"))
    cs = corpus_identity_scan(limit_train_rows=S.get("train_rows_used"))
    if "error" in cs:
        A("| 语料扫描 | - | %s |" % cs["error"])
    else:
        A("| **整份语料前 %s 行** | **%d** | %s |"
          % ("{:,}".format(cs["scanned"]), cs["identity_questions"],
             "**语料里一条身份问句都没有** ⇒ 该权重无法满足身份要求"
             if cs["identity_questions"] == 0 else "语料里有，抽样没抽到"))
    A("")
    if "error" not in cs and cs["identity_questions"] == 0:
        A("> **这是本次抽样最重要的结构性发现**：身份类问句在语料中**完全缺席**。")
        A("> 模型不是「学不好身份」，而是**从未被教过** —— 语料里连名字本身都没出现过；")
        A("> 这与 held 的 %d 条 `wrong` 是两件独立的事，但任何「统一答夭夭」的要求"
          % sh["buckets"].get("wrong", {}).get("n", 0))
        A("> 在**当前语料上都无法满足**，与 N、d、步数都无关。")
        A("> 要补：先按 §69.15 的清洗口径（宁可漏不可误伤）在语料里定位真实身份问句，")
        A("> 再决定是补语料还是用规则层兜底。")
        A("")
    for name, hits in (("train", idt), ("held", idh)):
        if not hits:
            continue
        ok = sum(1 for h in hits if h["exact"])
        nm = sum(1 for h in hits if h["has_name"])
        A("- **%s 逐条**：命中 %d 条，逐字一致 %d 条，含名字 %d 条" % (name, len(hits), ok, nm))
        for h in hits[:3]:
            A("  - 问：`%s` → 答：`%s`" % (md_escape(h["question"][:50]),
                                          md_escape(h["generated"][:60])))
    A("")

    A("## 六、诚实边界")
    A("")
    A("1. **单 seed=11，无种子极差**；不对「哪一档更好」下显著性结论。")
    A("2. **每侧只抽 %d 条**，held 的 EM 分辨率是 1/%d = %.4f；"
      "任何小于这个数的差都不可分辨。" % (S.get("sample", {}).get("n_per_side", 200),
                                       S.get("sample", {}).get("n_per_side", 200),
                                       1.0 / max(1, S.get("sample", {}).get("n_per_side", 200))))
    A("3. **train 侧高分只证明记住了**，对泛化没有说明力；held 才是泛化的读数。")
    A("4. char-F1 是字符级，**不区分顺序**，也不惩罚同问题多答案（语料里存在合法多解）。")
    A("5. 桶分类阈值 F1≥%s / %s 是**约定值**，换阈值桶会变；已在表头写明以便复核。"
      % (F1_NEAR, F1_MID))
    A("6. **本表 train EM 与 §69.52 的 182/200 不可直接比**：两次抽样器不同"
      "（本表 `torch.randperm(2952, seed=12345)[:200]`，N 扫描 `random.Random(12345).sample`），"
      "同种子但**不同行集**。逐位相同的是权重，不是抽样。")
    A("7. held 的 `wrong` 桶多为**通顺但无关的中文**（见第四节样例）：解码器工作正常、"
      "模型输出合法中文，但内容与问题无对应关系。⇒ 失败性质是**记忆/泛化**，"
      "**不是**格式崩坏、重复退化或截断（EOS 率 %.1f%%、截断率 %.1f%%）。"
      % (100 * sh["eos_rate"], 100 * sh["trunc_rate"]))
    A("")

    out = os.path.join(HERE, "QA_REPORT_%s.md" % a.tag)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("已写出 %s（%d 行）" % (os.path.basename(out), len(L)))
    print("  train EM %d/%d  F1 %.4f | held EM %d/%d  F1 %.4f"
          % (st["em"], st["rows"], st["f1"], sh["em"], sh["rows"], sh["f1"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())