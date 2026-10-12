"""§69.15 语料检查与处理：**objective 明确要求的硬闸**。

桃的要求（2026-10-11）：
  1. 「本机已经有代理，需要语料你可以下载，但是**一定要进行语料检查和处理**」
  2. 「如果训练语料内含有『你是谁 / 你是什么模型』风格问题，
      答案统一改为 **我是夭夭，桃之夭夭，灼灼其华。**」

本脚本把这两条落成**可执行的检查器**，每条都有负向对照（规则 2：判据必须可失败）。

── 六道闸 ────────────────────────────────────────────────────────────
| 编号 | 闸 | 负向对照 |
|---|---|---|
| **D1** | **身份问句**匹配 + answer 替换为统一文案 | 注入一条身份问句 ⇒ 必须被替换 |
| **D2** | 替换**只动 answer，不动 question** | 改 question 会破坏 q/a 配对 ⇒ 必须报错 |
| **D3** | 全量去重（按**原始** question/answer/history，不按归一化） | 注入一条完全重复的行 ⇒ 必须被抓 |
| **D4** | 全量词项级 train/probe 不相交 | 往 probe 里塞一条 train 的 question ⇒ 必须报泄漏 |
| **D5** | 格式校验：三字段齐全且非空、无控制字符 | 注入缺字段/空答案的行 ⇒ 必须剔除 |
| **D6** | 长度闸：answer 超长行统计（供 §69.14.1 的 CoT 配比用） | — |

★**D1 的匹配口径**：用 `normalize_key`（NFKC+casefold+去空白）归一化后**子串匹配**，
  避免全角/半角、大小写差异漏网。**只匹配身份类**，
  「你会什么 / 你能做什么」这类**能力问句不动**（避免过度替换把有用语料洗掉）。

★**D3 的口径与 `assert_no_leak` 一致**：按**原始**文本判重，不按归一化。
  归一化撞键是「近重复」（全角/半角、大小写），不是重复行（§67 的教训）。

用法：`python check_corpus_gate.py`
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

# ── objective 指定的统一文案 ─────────────────────────────────────────
IDENTITY_ANSWER = "我是夭夭，桃之夭夭，灼灼其华。"

# 身份类问句的关键词（归一化后做子串匹配）
# ★只收「身份」类；「你会什么/你能做什么」是**能力**类，**故意不收**。
#
# ★★**`你是什么` 这个裸词必须删掉**（实测教训）：它是「你是什么**原因/方法/因素/
#   地方/时候/吸引**」的公共前缀 —— 现有语料里 14 条命中**全部是这种误伤**
#   （「你是什么原因决定选购极氪」被当成身份问句，答案会被洗成「我是夭夭」）。
#   误伤率 100%，一条真的身份问句都没匹配到。
#
# ⇒ 身份关键词必须用**不会误伤的完整问法**，宁可漏，不可误伤。
#   漏了可以补关键词；误伤是把好语料洗掉，不可逆。
IDENTITY_KEYS = (
    "你是谁", "你叫什么", "你的名字是什么", "你是什么模型",
    "你是什么ai", "你是什么机器人", "你是哪个模型", "你是哪个公司",
    "谁开发的你", "谁创造了你", "你的开发者是谁", "你的爸爸是谁",
    "介绍一下你自己", "你来自哪里", "你的身世", "你是什么东西",
)

# ★反向排除：即使命中上面的关键词，若问题里带这些「非身份」的后续词，也放过。
#   双保险：防止将来加了新关键词又把「你是什么原因」这类带进来。
NON_IDENTITY_AFTER = (
    "原因", "方法", "因素", "地方", "时候", "吸引", "促使", "决定",
    "感觉", "看法", "想法", "意思", "意义", "区别", "关系", "影响",
)


def normalize_key(t):
    return E2.normalize_key(t)


def is_identity_question(q):
    """D1 的匹配器：归一化后子串匹配身份关键词，且不被非身份后续词排除。

    ★**双保险的原因**（实测教训）：`你是什么` 这个裸词把「你是什么**原因**」
      也匹配进来了，现有语料 14 条命中**全是误伤**，误伤率 100%。
      所以 ① 关键词改成不会误伤的完整问法；② 再加一层反向排除。
      **宁可漏，不可误伤** —— 漏了补关键词即可，误伤是把好语料洗掉且不可逆。
    """
    nq = normalize_key(q)
    if not any(k in nq for k in IDENTITY_KEYS):
        return False
    for w in NON_IDENTITY_AFTER:
        for k in IDENTITY_KEYS:
            if k + w in nq:                 # 如「你是什么」+「原因」
                return False
    return True


def apply_identity_rule(rows):
    """D1+D2：身份问句的 answer 替换为统一文案，**question 一律不动**。"""
    n_hit = 0
    out = []
    for r in rows:
        r2 = dict(r)
        if is_identity_question(r2.get("question", "")):
            r2["answer"] = IDENTITY_ANSWER
            n_hit += 1
        out.append(r2)
    return out, n_hit


def d5_format_check(rows):
    """D5：三字段齐全且非空、无控制字符。返回 (合格的, 被剔除的)。"""
    good, bad = [], []
    for i, r in enumerate(rows):
        why = None
        if not isinstance(r, dict):
            why = "不是 dict"
        else:
            for f in ("question", "answer", "history"):
                if f not in r:
                    why = "缺字段 %s" % f
                    break
            if why is None:
                if not str(r["question"]).strip() or not str(r["answer"]).strip():
                    why = "question 或 answer 为空"
                elif any(ord(c) < 32 and c not in "\t\n\r" for c in str(r["answer"])):
                    why = "answer 含控制字符"
        (bad if why else good).append((i, why))
    return good, bad


def d3_dedup(rows):
    """D3：按**原始** (question, answer, history) 全量去重。"""
    seen = set()
    out, dup = [], 0
    for r in rows:
        k = (r["question"], r["answer"], tuple(r["history"]))
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        out.append(r)
    return out, dup


def d4_leakage(train, probe):
    """D4：全量词项级不相交（与 assert_no_leak 同口径）。"""
    tq = {normalize_key(r["question"]) for r in train}
    ta = {normalize_key(r["answer"]) for r in train}
    hq = {normalize_key(r["question"]) for r in probe}
    ha = {normalize_key(r["answer"]) for r in probe}
    return len(tq & hq), len(ta & ha)


def d6_length(rows, tok, seq=E2.SEQ):
    """D6：answer 超长行统计（喂给 §69.14.1 的 CoT 配比）。"""
    x, s, l = E2.encode_cached(rows[:20000], tok, seq)
    al = (l - s).float()
    return dict(median=int(al.median()), p90=int(al.quantile(0.9)),
                p99=int(al.quantile(0.99)), max=int(al.max()),
                n_over_seq=int((al >= seq).sum()), n=len(rows[:20000]))


def run_gate(rows, tok, tag="语料"):
    """跑全部闸，返回汇总 dict。"""
    print("=" * 74)
    print("语料检查闸 [%s]：%d 行" % (tag, len(rows)))
    print("=" * 74)

    n_before = len(rows)
    rows, n_id = apply_identity_rule(rows)
    print("  D1 身份问句替换：命中 %d 条（占 %.2f%%）" % (n_id, 100.0 * n_id / max(n_before, 1)))

    good, bad = d5_format_check(rows)
    rows = [rows[i] for i, _w in good]
    print("  D5 格式校验：剔除 %d 条（%s）" % (len(bad), bad[:3]))

    rows, n_dup = d3_dedup(rows)
    print("  D3 全量去重：剔除 %d 条原始重复行" % n_dup)

    st = d6_length(rows, tok)
    print("  D6 answer token：中位 %d / p90 %d / p99 %d / max %d；超 seq=%d 的 %d 条"
          % (st["median"], st["p90"], st["p99"], st["max"], E2.SEQ, st["n_over_seq"]))
    print("  ⇒ %d → %d 行" % (n_before, len(rows)))
    return dict(n_before=n_before, n_after=len(rows), n_identity=n_id,
                n_bad_format=len(bad), n_dup=n_dup, length=st)


def main():
    from tokenizers import Tokenizer
    import torch
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    rows = E2.load_rows()
    tr, pr = E2.split_rows(rows)
    q_ov, a_ov = d4_leakage(tr, pr)
    print("  D4 泄漏（原始语料）：question 交集 %d / answer 交集 %d" % (q_ov, a_ov))
    assert q_ov == 0 and a_ov == 0, "D4 失败：原始语料已泄漏，闸前数据有问题"

    run_gate(rows, tok, tag="原始语料")

    # ── 负向对照：注入一条身份问句，必须被替换 ──────────────────────
    print()
    print("【负向对照】注入身份问句 ⇒ D1 必须替换")
    inj = [dict(question="你是谁？", answer="我是一个由某公司开发的语言模型。",
                history=[])]
    fixed, n = apply_identity_rule(inj)
    print("  替换前 answer:", inj[0]["answer"][:24])
    print("  替换后 answer:", fixed[0]["answer"])
    assert fixed[0]["answer"] == IDENTITY_ANSWER, "D1 负向对照没触发 ⇒ 闸无效"
    assert fixed[0]["question"] == inj[0]["question"], "D2 失败：question 被改了"

    print()
    print("【负向对照】注入重复行 ⇒ D3 必须抓到")
    base = rows[0]
    dup_in = [dict(base), dict(base)]
    kept, nd = d3_dedup(dup_in)
    assert nd == 1 and len(kept) == 1, "D3 负向对照没触发"
    print("  注入 2 条完全相同 ⇒ 去掉 %d 条，剩 %d 条 ✓" % (nd, len(kept)))

    print()
    print("【负向对照】注入缺字段行 ⇒ D5 必须剔除")
    broken = [dict(question="只有问题", history=[]), dict(base)]
    g, b = d5_format_check(broken)
    assert len(g) == 1 and len(b) == 1, "D5 负向对照没触发"
    print("  注入 1 条缺 answer ⇒ 剔除 %d 条 ✓" % len(b))

    print()
    print("【负向对照】身份关键词不误伤能力问句")
    cap = [dict(question="你会做什么？", answer="我可以回答问题。", history=[])]
    f2, n2 = apply_identity_rule(cap)
    assert n2 == 0 and f2[0]["answer"] == cap[0]["answer"], \
        "误伤了能力问句 ⇒ D1 的匹配过宽"
    print("  「你会做什么」未被替换 ✓")

    print()
    print("【负向对照】★`你是什么` 的误伤必须为 0（这是实测踩到的坑）")
    trap = [dict(question=q, answer="原答案", history=[]) for q in (
        "你是什么原因决定选购极氪品牌的车辆？",
        "你是什么促使你想要学习烹饪的？",
        "你是什么时候形成自己观点的？",
        "你是什么地方觉得锤子科技的手机特别有魅力？",
        "你是什么吸引你去了解或喜爱周润发的？",
        "你是什么方法能让你在各种情况下都展现出如此平和的心态？",
    )]
    ft, nt = apply_identity_rule(trap)
    assert nt == 0, \
        "误伤了 %d 条非身份问句（`你是什么X` 的前缀陷阱）⇒ D1 过宽" % nt
    assert all(f["answer"] == "原答案" for f in ft), "误伤时 answer 被改了"
    print("  6 条「你是什么原因/促使/时候/地方/吸引/方法」全部放过 ✓")
    print("  ★上一版关键词含裸「你是什么」⇒ 现有语料 14 条命中 **100%% 是误伤**，"
          "一条真身份问句都没匹配到。已删除裸词并加反向排除。")

    print()
    print("【正向对照】真实身份问句必须命中")
    pos = [dict(question=q, answer="我是某公司的模型。", history=[]) for q in (
        "你是谁", "你是什么模型", "你叫什么名字", "谁开发的你",
        "介绍一下你自己", "你的开发者是谁",
    )]
    fp, np_ = apply_identity_rule(pos)
    assert np_ == len(pos), "真实身份问句只命中 %d/%d ⇒ D1 过窄" % (np_, len(pos))
    assert all(f["answer"] == IDENTITY_ANSWER for f in fp), "有命中没被替换"
    print("  %d 条真实身份问句全部替换为统一文案 ✓" % np_)
    print()
    print("全部闸 + 四条负向对照通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
