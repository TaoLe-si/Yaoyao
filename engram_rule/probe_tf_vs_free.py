"""§69.41 教师强制 vs 自由生成的**第一步**落差：是分布偏移，还是根本没学会？

── 为什么做这个 ────────────────────────────────────────────────────
§69.40 用同源配对**第二次否掉**了「问题不可达」（不可见组 F1 只低 6.7%，
且可见组自己也只 0.2677 ⇒ 看得见也没用）。至此「第一个字为什么错」没有答案。

手上有一个没解释的落差：

| | 值 | 口径 |
|---|---:|---|
| `first_tok_acc` | **0.4043** | 教师强制，答案首 token 的 top-1 命中率（31,139 例） |
| 自由生成 LCP>0 | **36%** | 逐行 greedy 生成，首字符与参考答案一致的比例（200 例） |

**同一件事（第一个 token 对不对）在两个口径下差 4~12 个百分点**（且样本不同）。
本节把它们放到**同一批行、同一权重**上比，才能判断：

- 若教师强制对、自由生成错 ⇒ **分布偏移**（teacher forcing 与自回归的差异）
- 若两边都不对 ⇒ **模型根本没学会**这一步

── 判据（可失败，先登记）───────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **T1** | 同源配对成功（行数、种子、来源一致） | 不同源 ⇒ 拒绝配对（不做假对拍） |
| **T2** | 教师强制对的样本里，自由生成也占相当比例 | 若接近 0 ⇒ 强分布偏移；若接近 100% ⇒ 两者一致 |
| **T3** | 四格（TF对/自错、TF对/自对、TF错/自错、TF错/自对）都报 n | 有空格 ⇒ 该组合不存在，本身是信息 |

用法：`python probe_tf_vs_free.py`（一次前向 + 一次生成，约 3 分钟）
"""
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

CKPT = os.path.join(HERE, "engram_rule_scaling_v2_n124593_d128_wd0p0.pt")
QA = os.path.join(HERE, "qa_held_s200.jsonl")
REACH = os.path.join(HERE, "question_reach.json")
OUT = os.path.join(HERE, "tf_vs_free.json")
SAMPLE_SEED = 12345
SAMPLE = 200


def main():
    print("=" * 78)
    print("教师强制 vs 自由生成：第一步对不对（同源配对）")
    print("=" * 78)

    # ── T1 同源自检 ────────────────────────────────────────────────
    for p, nm in ((CKPT, "权重"), (QA, "qa_held_s200.jsonl"), (REACH, "question_reach.json")):
        if not os.path.exists(p):
            print("缺 %s：%s" % (nm, p))
            return 1
    qa = [json.loads(l) for l in open(QA, encoding="utf-8") if l.strip()]
    reach = json.load(open(REACH, encoding="utf-8"))
    if len(qa) != SAMPLE or reach.get("sample", {}).get("seed") != SAMPLE_SEED:
        print("T1 失败：qa 行数 %d ≠ %d，或 question_reach 种子 ≠ %d"
              % (len(qa), SAMPLE, SAMPLE_SEED))
        return 1
    print("  T1 同源自检通过（seed=%d, n=%d, held）" % (SAMPLE_SEED, len(qa)))

    import model as M                      # ★规则 7：E2 没导出 M
    E2.load_all()
    _rows = E2.load_rows()
    _tr, held_rows = E2.split_rows(_rows)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])                  # ★逐字用成品 cfg
    torch.manual_seed(E2.SEED)
    m = M.EngramRuleLM(M.RCfg(**cfg)).to(E2.DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    x, s, l = E2.encode_cached(_tr, E2._TOK, E2.SEQ)
    info = E2.DS.build_engram_from_examples(m, x, l)
    assert info["alive_slots"] > 0, "alive 全 False ⇒ 残壳"
    print("  建 Engram：alive %d / %d" % (info["alive_slots"], info["n_slots"]))

    # ★同一批行（与 qa / question_reach 同源）
    g = torch.Generator().manual_seed(SAMPLE_SEED)
    idx = torch.randperm(len(held_rows), generator=g)[:SAMPLE].tolist()
    samp = [held_rows[i] for i in idx]
    sx, ss, sl = E2.encode_cached(samp, E2._TOK, E2.SEQ)

    # ── 教师强制：答案首 token 的 top-1 是否命中 ────────────────────
    m.eval()
    tf_ok = []
    with torch.no_grad():
        for i in range(sx.shape[0]):
            bx = sx[i:i + 1].to(E2.DEV)
            st = int(ss[i])
            if st < 1 or st >= bx.shape[1]:
                tf_ok.append(False)
                continue
            logits = m(bx)[0, st - 1]        # 位置 st-1 预测位置 st
            tf_ok.append(int(logits.argmax().item()) == int(bx[0, st].item()))
    n_tf = sum(tf_ok)
    print()
    print("  教师强制第一步命中：%d / %d（%.1f%%）"
          % (n_tf, len(tf_ok), 100.0 * n_tf / len(tf_ok)))
    print("  （对照 §69.39 落盘的全量 31,139 例 first_tok_acc = 0.4043）")

    # ── 自由生成：首字符是否与参考答案一致 ──────────────────────────
    free_ok = []
    for j, r in enumerate(qa):
        a = r["reference"].strip()
        b = r["generated"].strip()
        free_ok.append(bool(a) and bool(b) and a[0] == b[0])

    # ── T3 四格 ────────────────────────────────────────────────────
    print()
    print("【四格（同源 %d 行）】" % len(qa))
    cells = {"TF对+自对": 0, "TF对+自错": 0, "TF错+自错": 0, "TF错+自对": 0}
    for t, f in zip(tf_ok, free_ok):
        k = ("TF对+自对" if t and f else "TF对+自错" if t else
             "TF错+自错" if not f else "TF错+自对")
        cells[k] += 1
    for k in ("TF对+自对", "TF对+自错", "TF错+自错", "TF错+自对"):
        print("  %-10s n=%-4d (%5.1f%%)" % (k, cells[k], 100.0 * cells[k] / len(qa)))
    assert sum(cells.values()) == len(qa), "T3 失败：四格之和 ≠ 总行数"

    # ── T2 判决 ────────────────────────────────────────────────────
    tf_and_free = cells["TF对+自对"] + cells["TF对+自错"]
    rate = cells["TF对+自对"] / max(tf_and_free, 1)
    print()
    print("=" * 78)
    print("【判决】")
    print("=" * 78)
    print("  教师强制第一步命中的 %d 条里，自由生成也命中的 = %d（%.1f%%）"
          % (tf_and_free, cells["TF对+自对"], 100 * rate))
    if rate < 0.5:
        print("  ⇒ **强分布偏移**：教师强制下第一步对，自由生成却不对。")
        print("     说明模型在「真值前缀」和「自己生成的前缀」上行为差异很大 ⇒")
        print("     这是 exposure bias，该用 scheduled sampling / 自我修正类手段，")
        print("     **不是加训练量、也不是改架构**。")
    elif rate > 0.9:
        print("  ⇒ 两者高度一致 ⇒ 没有明显分布偏移，第一步对错由模型本身决定。")
    else:
        print("  ⇒ 部分一致（%.1f%%）⇒ 有分布偏移成分，但模型本身也占一部分。" % (100 * rate))

    facts = dict(seed=SAMPLE_SEED, n=len(qa), tf_first_hit=n_tf,
                 tf_rate=n_tf / len(tf_ok), free_first_hit=sum(free_ok),
                 cells=cells, tf_to_free_rate=rate,
                 prior_full_first_acc=0.4043,
                 reach_cover_rate=reach.get("q_tail_cover_rate"))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  ★单 seed、200 条 ⇒ 无种子极差（规则 16）。")
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
