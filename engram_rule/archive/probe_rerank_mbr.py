"""§50: 生成后重排(MBR/rerank)能不能救"答非所问"。**只读, 不训练。**

候选池 = K 个采样(T=1.0, 固定种子) + 1 个 greedy。逐行打分:
  S_cond = mean_t log P(a_t | q, a_<t)             问题条件似然
  S_rel  = S_cond(q,a) - S_cond(q_null,a)          去偏(扣掉答案语言先验)
再与 oracle(候选里最好的) 和 greedy(现状) 对照。

**判决规则全部来自 DESIGN §50, 这里不新增判据。**
"""
import os
import sys
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
K = 8
TEMP = 1.0
GEN_SEED = 1234
NULL_Q = "请问"
HIT_F1 = 0.9
SE_MULT = 2.0


def mean_se(xs):
    n = len(xs)
    if n < 2:
        return sum(xs) / max(1, n), float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, (var / n) ** 0.5


def main():
    from tokenizers import Tokenizer
    from model import RCfg, EngramRuleLM
    import dialogue_small as ds
    from dialogue_small import split_records, char_f1
    import probe_delta_conditioning as pdc

    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    _, ev_rows, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                  seed=29, return_meta=True)
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # 1) 生成候选池 = 1 个 greedy + K 个采样
    # ★ §51 顺带修掉三处会让 §50 出错/不可复现的地方:
    #   (a) context_limit 原来吃默认值 96, 而 204 行 eval 里有 2 行 prompt>96 ⇒
    #       greedy_generate 会抛 "prompt exceeds context_limit"。改用 checkpoint 的
    #       真实 sequence_length(pos_abs_max)。
    #   (b) 8 个采样原来**共用一个 generator 顺序抽**, 于是第 k 个样本的随机数依赖
    #       前 k-1 个样本抽了几次 ⇒ 批化后(每步要交错着给 K 行抽)结果必然不同,
    #       等于把实验口径改了。现在改成**每个样本一个独立种子**(与顺序无关)。
    #   (c) ★temperature 是**批级参数**, greedy_generate_batch 里整批共用一个值。
    #       我第一版把 9 个候选一起批、TEMP=1.0 ⇒ **"greedy 基线"其实也是采样的**,
    #       基线被污染。修法: greedy 单独走(T=0), K 个采样单独走一批(T=TEMP)。
    #   (d) K 个采样共享同一行 ⇒ prompt 长度天然相等 ⇒ 可以整组批生成。
    ctx = int(ck["cfg"].get("pos_abs_max", SEQ))
    byte_map, _ = ds.bytelevel_token_bytes(tok)
    mask_cache = ds._DecodeMask(byte_map, tok.token_to_id("[EOS]"), DEV)
    pool = []
    for i, r in enumerate(ev_rows):
        greedy = ds.greedy_generate(model, tok, r, max_new_tokens=90, device=DEV,
                                    context_limit=ctx, mask_cache=mask_cache)["generated"]
        gens = [torch.Generator(device=DEV).manual_seed(GEN_SEED + i * 1000 + k)
                for k in range(K)]
        samples = [c["generated"] for c in
                   ds.greedy_generate_batch(model, tok, [r] * K, max_new_tokens=90,
                                            device=DEV, context_limit=ctx,
                                            temperature=TEMP, mask_cache=mask_cache,
                                            generators=gens)]
        pool.append([greedy] + samples)
        if (i + 1) % 40 == 0:
            print("  生成 %d/%d" % (i + 1, len(ev_rows)), flush=True)

    # 2) 打分(两个问题条件)
    flat, spans = [], []
    for i, cands in enumerate(pool):
        spans.append((len(flat), len(flat) + len(cands)))
        for c in cands:
            flat.append({"question": ev_rows[i]["question"], "answer": c, "history": []})
    _, s_cond = pdc.score_rows(model, flat, tok, SEQ)
    null_rows = [{"question": NULL_Q, "answer": r["answer"], "history": []} for r in flat]
    _, s_null = pdc.score_rows(model, null_rows, tok, SEQ)
    s_rel = s_cond - s_null

    # 3) 逐行汇总
    rows = []
    for i, r in enumerate(ev_rows):
        a, b = spans[i]
        cands = pool[i]
        f1s = [char_f1(c, r["answer"]) for c in cands]
        rows.append(dict(
            greedy_f1=f1s[0],
            oracle_f1=max(f1s),
            hit=float(max(f1s) >= HIT_F1),
            pick_cond=f1s[int(s_cond[a:b].argmax())],
            pick_rel=f1s[int(s_rel[a:b].argmax())],
            pick_like=f1s[int(s_cond[a:b].argmax())],   # 占位, 见下方说明
        ))

    def col(k):
        return [x[k] for x in rows]

    print("\n=== §50 重排判决 (eval %d 行, 候选 %d 个 = greedy+%d 采样) ==="
          % (len(rows), K + 1, K))
    keys = [("greedy_f1", "greedy (现状基线)"), ("oracle_f1", "oracle (候选里最好的)"),
            ("pick_cond", "pick(S_cond)"), ("pick_rel", "pick(S_rel)")]
    stat = {}
    print("  %-22s %10s %10s %10s" % ("指标", "mean", "SE", "mean>0.9"))
    for k, name in keys:
        m, s = mean_se(col(k))
        stat[k] = (m, s)
        print("  %-22s %10.4f %10.4f %10.4f" % (name, m, s, sum(col(k)) / len(rows)))
    h = sum(col("hit")) / len(rows)
    print("  %-22s %10.4f  ← 召回上限(存在 charF1>=%.1f 的候选的比例)"
          % ("hit@%d" % (K + 1), h, HIT_F1))

    print("\n=== 判决 ===")
    print("  oracle − greedy = %+.4f (2SE=%.4f)"
          % (stat["oracle_f1"][0] - stat["greedy_f1"][0],
             SE_MULT * ((stat["oracle_f1"][1] ** 2 + stat["greedy_f1"][1] ** 2) ** 0.5)))
    for k, name in [("pick_cond", "S_cond"), ("pick_rel", "S_rel")]:
        d = stat[k][0] - stat["greedy_f1"][0]
        se = (stat[k][1] ** 2 + stat["greedy_f1"][1] ** 2) ** 0.5
        verdict = ("无差异" if abs(d) <= SE_MULT * se
                   else ("**更好**" if d > 0 else "★**更差**(显著为负)"))
        print("  %s − greedy = %+.4f (2SE=%.4f) => %s" % (name, d, SE_MULT * se, verdict))
    d = stat["pick_rel"][0] - stat["pick_cond"][0]
    se = (stat["pick_rel"][1] ** 2 + stat["pick_cond"][1] ** 2) ** 0.5
    # ★符号: 我第一版只判 abs(d) > 2SE 就写"去偏版更好" —— 实测 d 是**负**的,
    #   方向判反了(与 §48 判决脚本同一个坑: 换判据要重推方向, 不能只看绝对值)。
    #   §50 预登记的原话是"去偏版更好"是个**假设**, 判决必须按符号说话。
    rel_verdict = ("两者无差异" if abs(d) <= SE_MULT * se
                   else ("去偏版**更好**" if d > 0 else "★去偏版**更差**(与假设相反)"))
    print("  S_rel − S_cond = %+.4f (2SE=%.4f) => %s"
          % (d, SE_MULT * se, rel_verdict))

    # ---- oracle 的形状: 候选到底"差多远" ----
    orc = sorted(col("oracle_f1"))
    n = len(orc)
    zero = sum(1 for v in orc if v <= 0.0)
    print("\n  oracle f1 形状: =0 的行 %d/%d (%.0f%%) | p50=%.3f p90=%.3f max=%.3f"
          % (zero, n, 100 * zero / n, orc[n // 2], orc[int(0.9 * n)], orc[-1]))
    for lo, hi, name in ((0.0, 0.3, "0~0.3"), (0.3, 0.6, "0.3~0.6"),
                         (0.6, 0.9, "0.6~0.9"), (0.9, 1.01, ">=0.9")):
        c = sum(1 for v in orc if lo <= v < hi)
        print("    oracle ∈ [%s): %3d 行 (%.1f%%)" % (name, c, 100 * c / n))

    print("\n  [读法] hit@%d = %.4f ⇒ %s" % (K + 1, h, "机制判死" if h == 0 else
                                              "机制未死"))
    print("  [读法] pick 明显低于 oracle ⇒ 上限够但打分函数认不出该选哪个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
