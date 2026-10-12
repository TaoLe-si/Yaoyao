"""只读探针: 检验"Engram 检索只依据末尾 1~2 元 token ID ⇒ 在需要问题条件的那个位置
实际上几乎无法区分问题"这一主张。

背景(读码得到的事实, 不是假设):
  - Engram 的查询向量 q = 该位置**结尾的 1..2 个 token 的嵌入之和**
    (model.py:173-192, ngram_query; v11 的 ngram_max=2)
  - slot_ids 用的是以位置 i 结尾的窗口 [x[i-1], x[i]] (model.py:159)
  - 预测第 t 个 token 用的是位置 t-1 的隐状态
    ⇒ **决定答案首 token 的那个位置, 其 Engram 查询只是"问题的最后 2 个 token"的函数**

要测的:
  Q1  204 条留出问题里, 末尾 2 元组/1 元组的重复度 —— 若是"吗？[EOS标记]"这类近乎常量的结尾,
      则该位置的检索键在不同问题之间几乎无差别。
  Q2  在该位置, 204 条问题检索到的**候选槽集合**有多少种不同的? 最集中的一种覆盖多少条?
  Q3  对照: 在答案中段取同样口径, 唯一性应显著更高 (证明这个指标会变、不是写死的)。
  Q4  输出头不绑权的影响面: 留出答案里有多少 token 在训练答案中从未出现过。

只读, 不写任何产物。
"""
import os
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd, cfg = ck["state_dict"], ck["cfg"]
    extra = ck["extra"]
    ev = ck["eval_tokens"]                      # (204, 512)
    starts = torch.tensor(extra["eval_answer_starts"], dtype=torch.long)
    lengths = torch.tensor(extra["eval_lengths"], dtype=torch.long)

    from model import RCfg, EngramMemory
    rc = RCfg(**cfg)
    em = EngramMemory(rc)
    em.load_state_dict({k[len("engram."):]: v for k, v in sd.items() if k.startswith("engram.")})

    # ---- Q1: 决定首 token 那个位置 = start-1; 它看到的是问题的最后 2 个 token ----
    pos = starts - 1                            # (204,)
    last1 = ev[torch.arange(ev.shape[0]), pos]
    last2 = ev[torch.arange(ev.shape[0]), pos - 1]
    c1, c2 = Counter(last1.tolist()), Counter(zip(last2.tolist(), last1.tolist()))
    print("[Q1] 决定答案首 token 的位置(start-1)所看到的末尾 token")
    print("     末 1 元共 %d 种 / 204 条；最常见的一种占 %d 条 (%.1f%%)"
          % (len(c1), c1.most_common(1)[0][1], 100 * c1.most_common(1)[0][1] / 204))
    print("     末 2 元共 %d 种 / 204 条；最常见的一种占 %d 条 (%.1f%%)"
          % (len(c2), c2.most_common(1)[0][1], 100 * c2.most_common(1)[0][1] / 204))
    print("     末 2 元 Top5:", [(str(k), v) for k, v in c2.most_common(5)])

    # ---- Q2: 该位置检索到的候选槽集合有多少种 ----
    ids = em.slot_ids(ev)                        # (B,L,R)
    B, L, R = ids.shape
    idx = torch.arange(B)
    cand = ids[idx, pos]                        # (204, R) = 4 个候选槽
    sets = [tuple(sorted(r.tolist())) for r in cand]
    cs = Counter(sets)
    print("\n[Q2] 该位置每条问题检索到的 %d 个候选槽: 共有 %d 种不同组合"
          % (R, len(cs)))
    print("     最集中的组合覆盖 %d / 204 条 (%.1f%%)" % (cs.most_common(1)[0][1],
                                                           100 * cs.most_common(1)[0][1] / 204))
    print("     Top5 组合覆盖条数:", [v for _, v in cs.most_common(5)])
    # 候选槽总去重数 vs 组合去重数: 说明"同槽不同组合"的程度
    print("     204 条去重后用到的不同**槽**个数 = %d / %d" % (len(set(cand.flatten().tolist())), R * 204))

    # ---- Q3: 对照 —— 答案中段同口径的唯一性 (负向对照: 证明该指标会变) ----
    mid = ((starts + lengths) // 2).clamp(max=L - 1)
    cand_mid = ids[idx, mid]
    sets_mid = [tuple(sorted(r.tolist())) for r in cand_mid]
    cs_mid = Counter(sets_mid)
    c1m = Counter(ev[idx, mid].tolist())
    print("\n[Q3] 对照(答案中段 start..end 的中点) —— 同样口径")
    print("     当前位置 token 种类 = %d / 204 (首 token 处只有 %d 种)"
          % (len(c1m), len(c1)))
    print("     候选槽组合种类 = %d / 204 (首 token 处只有 %d 种)"
          % (len(cs_mid), len(cs)))
    print("     => %s" % ("判据可失败: 中段唯一性显著更高, 首 token 处的集中不是指标写死"
                           if len(cs_mid) > len(cs) else "★判据不可失败, Q2 结论作废, 先读检查本身"))

    # ---- Q4: 输出头不绑权 -> 留出答案里有���少 token 训练时从未出现 ----
    tokvocab = ck["tokenizer"]["json"]
    raw = extra["raw_dialogue_rows"]
    print("\n[Q4] 留出答案的 token 可输出性(需要重建 train/eval 文本映射, 本探针跳过)")
    print("     raw_dialogue_rows = %d 条可用作后续核对" % len(raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())