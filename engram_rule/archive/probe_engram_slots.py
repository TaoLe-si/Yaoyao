"""只读探针: v11 的 Engram 实际用了多少槽位 —— 关系到"4096 槽能否承载 820 条问答事实"。

背景: Engram 的 key/val 是**可训练 Embedding**(n_slots × d), alive 由**训练语料的 n-gram
哈希命中次数 >= min_count** 决定。v11: n_slots=4096, ngram_max=2, n_hash=2, min_count=1。

要回答:
  Q1  checkpoint 里 alive 的槽位有多少个 (出厂统计的结果)?
  Q2  820 条训练记录实际命中多少个不同槽位? 每槽平均被几条记录命中?
  Q3  一个位置只有 R = ngram_max * n_hash = 4 个候选槽 —— 存活率低不低?

负向对照: Q2 若换个输入必然变 (换一批 token); 且 Q2/Q1 不等号相同时必须能解释。
"""
import os
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    cfg = ck["cfg"]
    alive = sd["engram.alive"].bool()
    print("[Q1] checkpoint 内 alive 槽位 = %d / %d (存活率 %.3f)"
          % (int(alive.sum()), alive.numel(), float(alive.float().mean())))
    print("     n_slots=%d  ngram_max=%d  n_hash=%d  min_count=%d  R(每位置候选)=%d"
          % (cfg["n_slots"], cfg["ngram_max"], cfg["n_hash"], cfg["min_count"],
             cfg["ngram_max"] * cfg["n_hash"]))

    corpus = ck.get("corpus", {})
    print("\n     checkpoint corpus 元信息 keys:", list(corpus.keys())[:12])

    tok = ck.get("tokenizer", {})
    print("     tokenizer keys:", list(tok.keys())[:12])

    # 用保存的 eval_tokens 做一次真实数据上的槽位占用统计 (与训练同分布、同 tokenizer)
    ev = ck.get("eval_tokens")
    if ev is None:
        print("\n[Q2] checkpoint 未保存 eval_tokens, 跳过 (不臆造训练 token)")
        return 0
    print("\n[Q2] eval_tokens shape =", tuple(ev.shape))

    from model import RCfg, EngramMemory
    rc = RCfg(**cfg)
    em = EngramMemory(rc)
    em.load_state_dict({k[len("engram."):]: v for k, v in sd.items() if k.startswith("engram.")})

    ids = em.slot_ids(ev)                      # (B,L,R)
    hit = ids[alive[ids]]                      # 只看存活槽的命中
    uniq = torch.unique(hit)
    counts = Counter(hit.reshape(-1).tolist())
    per_slot = torch.tensor(list(counts.values()), dtype=torch.float32)
    print("     样本内命中存活槽的总次数 = %d" % hit.numel())
    print("     样本内出现的不同存活槽   = %d" % uniq.numel())
    print("     每槽命中次数 mean=%.2f  median=%.1f  max=%d"
          % (float(per_slot.mean()), float(per_slot.median()), int(per_slot.max())))

    # Q3: 每位置的 4 个候选里, 平均几个是存活的 —— 这是检索的实际信噪比
    live_cnt = alive[ids].sum(-1).float()
    print("\n[Q3] 每位置 R=4 个候选里的存活个数: mean=%.3f  ==0 的位置占比=%.3f"
          % (float(live_cnt.mean()), float((live_cnt == 0).float().mean())))
    print("     (==0 的位置走残差捷径: 注意力与 direct 支路全为 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())