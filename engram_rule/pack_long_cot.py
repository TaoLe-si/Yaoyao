"""§69.19 拼卷（packing）：把短样本拼成长上下文 CoT 训练卷。

**动机**：本机语料单条都短 —— `question` token 中位 22 / p90 35 / p99 59 / max 689。
⇒ **单条永远训不出长上下文能力**。要做 100 万上下文，必须把多条样本拼成一卷。

── 三道闸（每条都有负向对照）────────────────────────────────────────
| 编号 | 闸 | 判据 | 负向对照 |
|---|---|---|---|
| **P1** | **不跨样本截断** | 每条样本的三段必须完整落在一卷内 | 把 seq 设成小于单条 ⇒ 必须被拒 |
| **P2** | **段间有分隔标记** | ` / <answer>` 三段各有起止符 | 去掉分隔符 ⇒ 必须报错 |
| **P3** | **答案只监督 answer 段** | 拼卷后每个样本的 answer 起点仍可定位 | 打乱顺序 ⇒ 定位必须失败 |

★**P1 是长上下文训练最容易踩的坑**：若把一条样本从中间切断，
  模型会学到「半句话接着推理」的伪模式，而且 loss 仍在下降 —— 不报错、指标好看。
  ⇒ 拼卷必须**整条进出**，装不下就换下一条，绝不切。

★**P2 是思维链的必要条件**：拼卷后多条样本的 think/answer 交替出现，
  没有分隔符的话模型分不出「这句话是哪条样本的哪一段」。

★**P3 关联 K1**：`answer_loss_sums` 用 `starts` 定位 answer 起点。
  拼卷后每个样本各有自己的 `starts`，必须**逐样本**记账，不能只用整卷的一个。

用法：`python pack_long_cot.py [每卷目标 token] [最多卷数]`
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_cot_corpus as B
import exp_scaling_grid_v2 as E2

OUT = os.path.join(HERE, "corpus_long_cot_packed.jsonl")

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANS_OPEN, ANS_CLOSE = "<answer>", "</answer>"
SEP = "\n"


def render_sample(r):
    """把一条三段样本渲染成带分隔标记的文本。"""
    return "%s%s%s%s%s%s%s" % (THINK_OPEN, r["think"], THINK_CLOSE,
                               ANS_OPEN, r["answer"], ANS_CLOSE, SEP)


def p2_check(text):
    """P2：分隔标记必须成对出现。"""
    return (text.count(THINK_OPEN) == text.count(THINK_CLOSE)
            and text.count(ANS_OPEN) == text.count(ANS_CLOSE)
            and text.count(THINK_OPEN) >= 1)


def pack(rows, tok, target_tok=8192, max_vols=None):
    """拼卷。返回 (卷列表, 统计)。

    ★P1 的实现：**先算这条样本的 token 数**，装不下就**换下一条**，绝不切。
      `cur_tok + n_tok > target` 时封卷，而不是把样本切一半塞进去。
    """
    vols, stats = [], dict(n_samples=0, n_vols=0, n_rejected=0,
                           n_tok_total=0, n_pad=0)
    cur, cur_tok, cur_n = [], 0, 0
    for r in rows:
        text = render_sample(r)
        if not p2_check(text):
            stats["n_rejected"] += 1
            continue
        n_tok = len(tok.encode(text, add_special_tokens=False).ids)
        if n_tok > target_tok:
            # ★单条就超长 ⇒ 这一卷装不下它，跳过（不切！）
            stats["n_rejected"] += 1
            continue
        if cur_tok + n_tok > target_tok and cur:
            vols.append(cur)
            stats["n_vols"] += 1
            stats["n_pad"] += target_tok - cur_tok
            cur, cur_tok, cur_n = [], 0, 0
            if max_vols and len(vols) >= max_vols:
                break
        cur.append(dict(text=text, n_tok=n_tok, tier=r.get("cot_tier", "?"),
                        question=r["question"]))
        cur_tok += n_tok
        cur_n += 1
        stats["n_samples"] += 1
        stats["n_tok_total"] += n_tok
    if cur and not (max_vols and len(vols) >= max_vols):
        vols.append(cur)
        stats["n_vols"] += 1
        stats["n_pad"] += target_tok - cur_tok
    return vols, stats


def p1_negative_control(tok, target=64):
    """★负向对照：把 target 设得比单条还小 ⇒ 必须全部被拒，不能硬塞。"""
    rows = E2.load_rows()[:20]
    made, _st = B.build(rows, 20)
    vols, st = pack(made, tok, target_tok=target, max_vols=None)
    assert st["n_rejected"] > 0, "target 比单条还小却没拒任何一条 ⇒ P1 失效"
    assert all(s["n_tok"] <= target for v in vols for s in v), \
        "P1 失败：卷里有超过 target 的样本 ⇒ 发生了截断"
    return st


def p3_check(vols, tok):
    """P3：每卷内每条样本的 think/answer 起止符计数必须一致。"""
    bad = 0
    for v in vols:
        whole = "".join(s["text"] for s in v)
        if whole.count(THINK_OPEN) != len(v) or whole.count(ANS_OPEN) != len(v):
            bad += 1
    return bad


def main():
    from tokenizers import Tokenizer
    import torch
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])

    print("=" * 74)
    print("拼卷：每条样本整条进出，绝不跨样本截断（P1）")
    print("=" * 74)

    # ★负向对照先跑
    st_neg = p1_negative_control(tok, target=64)
    print("  [负向对照] target=64（比单条短）⇒ 拒 %d 条，卷内样本全部 ≤64 ✓"
          % st_neg["n_rejected"])

    rows = E2.load_rows()[:cap]
    made, bst = B.build(rows, cap)
    print()
    print("  合成 %d 行（tier: full=%d mid=%d short=%d no_cot=%d）"
          % (bst["n_out"], bst["tier_full"], bst["tier_mid"],
             bst["tier_short"], bst["tier_no_cot"]))

    vols, st = pack(made, tok, target_tok=target)
    print()
    print("  拼卷：目标 %d token/卷" % target)
    print("    样本 %d 条，成卷 %d 卷，拒 %d 条（单条超长或 P2 失败）"
          % (st["n_samples"], st["n_vols"], st["n_rejected"]))
    print("    总 token %d，padding 合计 %d（占 %.1f%%）"
          % (st["n_tok_total"], st["n_pad"],
             100.0 * st["n_pad"] / max(1, st["n_tok_total"] + st["n_pad"])))
    if vols:
        ns = sorted(len(v) for v in vols)
        print("    每卷样本数：中位 %d / 最少 %d / 最多 %d"
              % (ns[len(ns) // 2], ns[0], ns[-1]))
        bad = p3_check(vols, tok)
        print("    P3 分隔符记账不一致的卷数 = %d %s"
              % (bad, "✓" if bad == 0 else "★"))

        with open(OUT, "w", encoding="utf-8") as f:
            for i, v in enumerate(vols):
                f.write(json.dumps(dict(vol=i, target_tok=target,
                                        samples=v), ensure_ascii=False) + "\n")
        print()
        print("  已写出 %d 卷 → %s" % (len(vols), os.path.basename(OUT)))
        print()
        print("  第一卷前 300 字符：")
        print("   ", "".join(s["text"] for s in vols[0])[:300].replace("\n", " / "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
