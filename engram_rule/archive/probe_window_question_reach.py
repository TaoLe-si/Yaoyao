"""★问题跑偏的三个竞争解释 —— 一个都不测"对不对", 只测"模型到底有没有**看到**问题"。

**现象**(§60): d=128 在长问题上跑偏, 且跑偏点是**具体名词**(句法正常)。
桃问「目前架构的解码是怎样的, 评估是否需要引入全局注意力」。

**先把架构的信息通路讲清楚(零假设, 直接读代码)**:

  每个位置 t 的输出只可能来自三个来源:
    (a) **规则/窗口通路** —— 偏移 `off ∈ [start_bucket·step, +W)`, 长度恒为 `window_max=16`
    (b) **§4.4 direct 通路** —— `h += direct(agg)`, `agg` 来自 Engram 的 n-gram 检索, **不过窗口**
    (c) **绝对位置特征** —— `pos_bucket`, 只进桶偏置(即只影响"看哪里")

  ⇒ 问题如果**不在这 16 个 token 内、也不在某个 n-gram 的 key 里**, 就**完全不可达**。
     这与"注意力分数低"是两回事 —— 是**结构上不可达**, 不是"没被重视"。

**三个竞争假设**(跑之前写死, 每个都附"反例下会变"的证明):

  P1  **窗口看不见问题**: 当前步的窗口 `[off0, off0+W)` **不含**问题末尾
      ⇒ 若成立, 把 prompt 里的问题挪近答案(缩短 history)应显著改善
  P2  **窗口看得见问题, 但被后续 tokens 挤掉**: off0 落在中间, 问题在窗口外
      ⇒ 若成立, `last_startb` 会随答案变长而单调右移, 问题被挤出
  P3  **Engram 通路把问题做了索引, 但与位置无关**(§53 C4 已验证槽与位置无关)
      ⇒ 若成立, 问题必须能靠 n-gram 命中, 长问题里关键词出现的 2-gram 密度是关键

**判据可失败性**: 每条都配**负向对照**(改一个已知会翻转的量, 读数必须跟着翻)。
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIB = 1 << 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="取多少行(取长问题优先)")
    ap.add_argument("--min-gold", type=int, default=90, help="只看 gold_len 这么大的行")
    a = ap.parse_args()

    import exp_capacity_scaling as ecs
    from tokenizers import Tokenizer
    from model import RCfg, EngramRuleLM
    from dialogue_small import (format_prefix, format_record, greedy_generate,
                                split_records)

    ckpt = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_d128_s11.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr_rows, ev_rows, _ = split_records(ck["extra"]["raw_dialogue_rows"],
                                        train_fraction=0.8, seed=29, return_meta=True)
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cfg = ck["cfg"]
    W = cfg["window_max"]
    NB = cfg["n_buckets"]
    STEP = cfg["bucket_step"]
    print("d=%d  window_max=%d  n_buckets=%d  bucket_step=%d  pos_abs_max=%d  "
          "ngram_max=%d  n_hash=%d  n_slots=%d"
          % (cfg["d"], W, NB, STEP, cfg["pos_abs_max"], cfg["ngram_max"],
             cfg["n_hash"], cfg["n_slots"]))
    print("★架构上界(静态, 不依赖任何权重): 回看 ≤ (n_buckets-1)·step + W-1 = %d"
          % ((NB - 1) * STEP + W - 1))
    print()

    # 取长问题里最长的 n 行
    cand = [r for r in tr_rows
            if len(tok.encode(format_record(r), add_special_tokens=False).ids)
            - len(tok.encode(format_prefix(r), add_special_tokens=False).ids)
            >= a.min_gold]
    cand.sort(key=lambda r: -len(r["answer"]))
    rows = cand[:a.n]
    print("取 %d 行(全部 gold_len ≥ %d)" % (len(rows), a.min_gold))
    print()

    stats = dict(hit_window=0, miss_window=0, probe_in_win=0, probe_out_win=0,
                 hist_w=0, q_tok=0, drop_q=0, drop_rand=0, pad_changed=0, n=0,
                 bigpad_n=0, bigpad_in=0, bigpad_out=0, bigpad_gained=0,
                 bigpad_changed=0)
    detail = []
    detail2 = []
    for r in rows:
        prompt_ids = tok.encode(format_prefix(r), add_special_tokens=False).ids
        ref_ids = tok.encode(format_record(r), add_special_tokens=False).ids
        gold = ref_ids[len(prompt_ids):]
        if len(prompt_ids) < 4 or len(gold) < 8:
            continue
        stats["n"] += 1
        # ── 教师强制跑一遍, 拿到每步的 last_startb / last_window_off ────────
        x = torch.tensor([ref_ids], dtype=torch.long, device=DEV)
        for blk in model.blocks:
            blk.attn.record = True
        with torch.no_grad():
            model(x)
        for blk in model.blocks:
            blk.attn.record = False
        off0 = model.blocks[0].attn.last_window_off[0, -1].tolist()     # (W,)
        startb = int(model.blocks[0].attn.last_startb[0, -1])
        a0 = int(off0[0])
        # 当前步的窗口覆盖的**答案段内**相对偏移: 位置 t 的后 W 个 token
        # 问: 窗口 [a0, a0+W) 里有没有包含"问题末尾"的 token?
        # 问题末尾在整条序列里的绝对位置 = len(prompt_ids)-1
        # 答案起点 = len(prompt_ids)。设当前步 = 答案起点(即答案第 1 个 token 处),
        # 窗口覆盖的绝对位置 = [len(prompt_ids)-a0, len(prompt_ids)-a0+W)
        win_lo = len(prompt_ids) - a0
        win_hi = win_lo + W
        q_end = len(prompt_ids) - 1                    # 问题最后一个 token 的位置
        in_win = win_lo <= q_end < win_hi
        stats["probe_in_win" if in_win else "probe_out_win"] += 1
        detail.append((r, q_end, win_lo, win_hi, a0, startb, len(prompt_ids), len(gold)))

        # ── P1/P2 负向对照: 只改问题**长度**(在其前面垫字), 不删 history ────
        #   ★不用"删 history"做对照: 本批 20 行的 history 本来就是**空**的
        #     (`format_prefix` 里 `row["history"]` 为空 ⇒ 去掉它 prompt 一字不变) ⇒
        #     对照是空集, 判据恒真(第一版就踩了这个坑, 报出"可比较的行 0")。
        #   改成**在问题前垫固定前缀字**, 把问题末尾往后推 —— 长度变了、问题内容没变。
        #   若问题内容真有用, 推远之后答案质量应该**变差**。
        pad = "请问一下"
        r_pad = dict(question=pad + r["question"], answer=r["answer"], history=[])
        p_pad = tok.encode(format_prefix(r_pad), add_special_tokens=False).ids
        if len(p_pad) > len(prompt_ids) and (len(p_pad) - len(prompt_ids)) <= 8:
            stats["drop_q"] += 1
            q_end_p = len(p_pad) - 1
            win_lo_p = len(p_pad) - a0
            in_pad = win_lo_p <= q_end_p < win_lo_p + W
            if in_pad and not in_win:
                stats["hit_window"] += 1      # 垫字把问题推进窗口了
            elif in_win and not in_pad:
                stats["hist_w"] += 1          # 反而被挤出
            # 乱序对照: 同样长度, 但把问题字符打乱 ⇒ 内容坏了、长度没变
            if len(r["question"]) > 4:
                stats["q_tok"] += 1
                chars = list(r["question"])
                torch.manual_seed(0)
                sh = "".join(chars[i] for i in torch.randperm(len(chars)).tolist())
                r_sh = dict(question=pad + sh, answer=r["answer"], history=[])
                p_sh = tok.encode(format_prefix(r_sh), add_special_tokens=False).ids
                if len(p_sh) == len(p_pad):
                    g_pad = greedy_generate(model, tok, r_pad, max_new_tokens=90,
                                            device=DEV)
                    g_sh = greedy_generate(model, tok, r_sh, max_new_tokens=90,
                                           device=DEV)
                    if g_sh["generated"].strip() != g_pad["generated"].strip():
                        stats["drop_rand"] += 1
                    f_pad = g_pad["generated"].strip()
                    f_base = None
                    g_base = greedy_generate(model, tok, r, max_new_tokens=90, device=DEV)
                    f_base = g_base["generated"].strip()
                    if f_pad != f_base:
                        stats["pad_changed"] += 1

    # ── 强对照: 垫足够多的字把问题真正推进窗口(4 个字推不动) ─────────────
    #   ★为什么要单独做: 窗口 off0 只取 {0, 32, …}(start_bucket·step, step=32),
    #     而窗口长 16 ⇒ 窗口是 [.., ..+16) 的**离散区间**, 垫 4 个字根本跳不进
    #     下一个区间。必须一次垫过 step(32)才有机会。
    for r, q_end, win_lo, win_hi, a0, startb, pl, gl in detail:
        in_win = win_lo <= q_end < win_hi
        # 目标: 让 q_end 落在 [win_lo+32, win_lo+32+16) —— 即起点跳到下一个桶
        target_lo = win_lo + STEP
        pad_needed = max(0, target_lo - q_end)
        big_pad = max(STEP, ((pad_needed + STEP - 1) // STEP) * STEP)
        r_big = dict(question=("请" * big_pad) + r["question"], answer=r["answer"],
                     history=[])
        p_big = tok.encode(format_prefix(r_big), add_special_tokens=False).ids
        # 只接受"长度增量在预期内"的行(避免 tokenizer 边界效应把长度改太多)
        delta = len(p_big) - pl
        if not (big_pad * 0.6 <= delta <= big_pad * 1.4):
            continue
        stats["bigpad_n"] += 1
        q_end_b = len(p_big) - 1
        win_lo_b = len(p_big) - a0
        in_big = win_lo_b <= q_end_b < win_lo_b + W
        stats["bigpad_in" if in_big else "bigpad_out"] += 1
        if in_big and not in_win:
            stats["bigpad_gained"] += 1
        g_big = greedy_generate(model, tok, r_big, max_new_tokens=90, device=DEV,
                                context_limit=600)
        g_base = greedy_generate(model, tok, r, max_new_tokens=90, device=DEV,
                                 context_limit=600)
        if g_big["generated"].strip() != g_base["generated"].strip():
            stats["bigpad_changed"] += 1
        detail2.append((big_pad, not in_win, not in_big,
                        g_base["generated"].strip()[:18],
                        g_big["generated"].strip()[:18]))

    print("=" * 74)
    print("A. 首步窗口覆盖(只统计第 2~3 层即可, 第 0 层 startb 有代表性)")
    print("=" * 74)
    print("  n=%d  问题末尾在窗口内 %d / 窗口外 %d"
          % (stats["n"], stats["probe_in_win"], stats["probe_out_win"]))
    print("  ⇒ %s"
          % ("★P1 可能成立: 答案第 1 步就看不见问题" if stats["probe_out_win"] > stats["n"] * 0.5
             else "问题在窗口内 ⇒ P1 在**首步**不成立"))
    print()

    print("=" * 74)
    print("B. 逐行: 窗口 [win_lo, win_hi) 与问题末尾 q_end")
    print("=" * 74)
    print("  %-6s %-6s %-8s %-8s %-6s | %-6s %-6s %-10s"
          % ("q_end", "win_lo", "win_hi", "off0", "起桶", "prompt", "gold", "判定"))
    for r, q_end, lo, hi, a0, sb, pl, gl in detail[:a.n]:
        print("  %-6d %-6d %-8d %-8d %-6d | %-6d %-6d %-10s"
              % (q_end, lo, hi, a0, sb, pl, gl,
                 "窗内" if lo <= q_end < hi else "★窗外"))
    print()

    print("=" * 74)
    print("C. 负向对照: 在问题前垫 4 个字(内容不变, 只把问题往后推)")
    print("=" * 74)
    print("  可比较的行 %d ; 其中 %d 行问题被推进窗口 ; %d 行反而被挤出"
          % (stats["drop_q"], stats["hit_window"], stats["hist_w"]))
    print("  垫字后生成改变的行: %d / %d" % (stats["pad_changed"], stats["drop_q"]))
    print("  乱序问题(长度/结构不变)后生成改变的行: %d / %d"
          % (stats["drop_rand"], stats["q_tok"]))
    print()
    print("  ★判据1: 垫字后生成**必须变** ⇒ 否则模型在按长度/结构猜, 与问题内容无关")
    print("  ★判据2: 乱序后生成改变的行数 vs 垫字的 ⇒ 乱序改变更少 ⇒ 内容敏感度低")
    print("  ★判据3: 若垫字把问题推进了窗口、且生成改变 ⇒ 位置是瓶颈(P1 加强)")
    print()

    print("=" * 74)
    print("D. ★★强对照: 垫**足够多**的字, 把问题真正推进窗口再生成")
    print("=" * 74)
    print("  可比较 %d 行 ; 进窗口 %d / 仍在窗口外 %d ; 其中新增进窗口 %d"
          % (stats["bigpad_n"], stats["bigpad_in"], stats["bigpad_out"],
             stats["bigpad_gained"]))
    print("  大垫字后生成改变的行: %d / %d"
          % (stats["bigpad_changed"], stats["bigpad_n"]))
    print()
    if detail2:
        print("  %-6s %-7s %-7s | %-20s | %-20s"
              % ("垫字数", "原窗外", "现窗外", "原生成", "垫后生成"))
        for bp, w0, w1, g0, g1 in detail2[:12]:
            print("  %-6d %-7s %-7s | %-20s | %-20s"
                  % (bp, "是" if w0 else "否", "是" if w1 else "否", g0, g1))
    print()
    print("  ★★★判决判据: 若「问题进窗口后生成**仍不变**」⇒ **窗口位置不是瓶颈**,")
    print("      问题通路依赖的是 Engram 的 n-gram 检索, 不是窗口覆盖。")
    print("      这是 P3 成立、P1/P2 否证的决定性读数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
