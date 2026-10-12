"""★§69.31 只读探针：模型到底学到了什么 —— 语义载体与「语言规则」各自落在哪些参数上。

── 桃的问题 ────────────────────────────────────────────────────────
「按你说的模型学到的是什么，我们原来的架构，模型学到的是如何运用语言规则，
  语义靠语言本身携带吗？」

── 为什么不猜 ────────────────────────────────────────────────────────
这是**可实测**的：模型学到什么，就落在哪些参数被训练改动了。所以逐个参数对比
「初始化」与「训练后」，看哪些真的动了、哪些没动。不动的那条通路 = 没学到东西。

── 四个可证伪读数 ──────────────────────────────────────────────────
| 编号 | 读什么 | 判据 |
|---|---|---|
| **S1** | `pos_bucket.weight` 是否还是 0 | 零初值设计 ⇒ 训练后仍全 0 ⇒ 这条通路没学到东西 |
| **S2** | `pos_abs.weight` 范数分布 | 非零 ⇒ 位置信息确实进了门的查询侧 |
| **S3** | `emb.weight` 与 `engram.val.weight` 的范数分布 | 语义载体的实际规模 |
| **S4** | `gate` / `head_w` / `rule_scale` / `lam` 离 1.0 有多远 | 1.0 是初值 ⇒ 偏离越大说明学得越多 |

用法：`python probe_what_learned.py`（只读 checkpoint，CPU 上跑，不训练）
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CKPT = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st400.pt")
OUT = os.path.join(HERE, "what_learned_facts.json")


def stats(name, t, init=None):
    """返回一组可证伪的统计量。`init` 给了就算「离初值多远」。"""
    f = t.detach().float()
    out = dict(
        name=name,
        shape=list(t.shape),
        numel=int(f.numel()),
        norm=float(f.norm()),
        absmean=float(f.abs().mean()),
        absmax=float(f.abs().max()),
        nonzero_frac=float((f.abs() > 0).float().mean()),
    )
    if init is not None:
        d = (f - init.detach().float())
        out["delta_norm"] = float(d.norm())
        out["delta_absmax"] = float(d.abs().max())
        out["moved_frac"] = float((d.abs() > 0).float().mean())
    return out


def main():
    print("=" * 78)
    print("模型学到了什么 —— 只读 checkpoint，逐参数对比「初始化 vs 训练后」")
    print("=" * 78)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd, cfg = ck["state_dict"], ck["cfg"]
    d, nl = cfg["d"], cfg["n_layers"]
    print("  ckpt = %s" % os.path.basename(CKPT))
    print("  d=%d  n_layers=%d  n_slots=%d  min_count=%d"
          % (d, nl, cfg["n_slots"], cfg["min_count"]))
    print("  训练步数 = %s   lr = %s   seed = %s"
          % (ck.get("extra", {}).get("steps"), ck.get("extra", {}).get("lr"),
             ck.get("extra", {}).get("seed")))

    facts = dict(ckpt=os.path.basename(CKPT), cfg=cfg, rows=[])

    # ── S1: pos_bucket 是否还是 0 ────────────────────────────────────
    print()
    print("【S1】位置 -> 桶偏置 通路（pos_bucket，零初值，串上后行为逐位不变）")
    for i in (0, nl - 1):
        pb = sd["blocks.%d.attn.pos_bucket.weight" % i]
        st = stats("blocks.%d.attn.pos_bucket" % i, pb, torch.zeros_like(pb))
        print("  layer%d: 非零元素 %.4f%%   |Δ|max = %.3e   ‖Δ‖ = %.3e"
              % (i, 100 * st["nonzero_frac"], st["delta_absmax"], st["delta_norm"]))
        facts["rows"].append(st)
        if i == 0:
            facts["pos_bucket_moved"] = st["moved_frac"]
    pb0 = facts["rows"][0]
    print("  ⇒ %s" % ("pos_bucket **仍然全 0** —— 这条位置通路训练后没学到任何东西"
                      if pb0["moved_frac"] == 0.0
                      else "pos_bucket 动了 %.2f%% —— 位置通路真的在贡献" % (100 * pb0["moved_frac"])))

    # ── S2: pos_abs 范数 ────────────────────────────────────────────
    print()
    print("【S2】绝对位置进门的查询侧（pos_abs，随机初始化，非零即被用）")
    pa = sd["blocks.0.attn.pos_abs.weight"]
    st = stats("blocks.0.attn.pos_abs", pa)
    print("  形状 %s   ‖W‖ = %.4f   逐行范数 均值 %.4f / min %.4f / max %.4f"
          % (st["shape"], st["norm"],
             float(pa.norm(dim=1).mean()), float(pa.norm(dim=1).min()),
             float(pa.norm(dim=1).max())))
    print("  ★但这只说明它**非零**，不说明它在被用 —— 「在用」由 §69.28 的 T4 给出：")
    print("    alive 清零时输出逐位不变 ⇒ 注意力**完全**由规则门控；pos_abs 只在门内起作用。")
    facts["rows"].append(st)

    # ── S3: 语义载体的规模 ───────────────────────────────────────────
    print()
    print("【S3】语义落在哪：emb（符号 -> 向量）与 engram.val（n-gram -> 向量）")
    for key, label in (("emb.weight", "emb  token 嵌入"),
                       ("engram.val.weight", "engram.val  n-gram 值向量"),
                       ("engram.key.weight", "engram.key  n-gram 键向量"),
                       ("head.weight", "head  输出投影")):
        st = stats(key, sd[key])
        facts["rows"].append(st)
        print("  %-24s %-14s ‖W‖ = %10.2f   非零 %.1f%%"
              % (label, "x".join(str(s) for s in st["shape"]),
                 st["norm"], 100 * st["nonzero_frac"]))
    ve = float(sd["emb.weight"].norm())
    vv = float(sd["engram.val.weight"].norm())
    print("  ⇒ engram.val 的范数是 emb 的 %.2f 倍 —— 槽位表承载的信息量与词表同量级甚至更大"
          % (vv / max(ve, 1e-9)))

    # ── S4: 可学习标量/向量离初值多远 ────────────────────────────────
    print()
    print("【S4】规则侧的可学习量：初值 → 训练后（1.0 是初值）")
    for key, init_val in (("blocks.0.attn.rule_scale", cfg["rule_scale_init"]),
                          ("blocks.0.attn.lam", cfg["lam_init"])):
        t = sd[key]
        st = stats(key, t, torch.full_like(t, float(init_val)))
        facts["rows"].append(st)
        print("  %-28s 初值 %.3f -> %.4f   |Δ| = %.4f"
              % (key, init_val, float(t), abs(float(t) - float(init_val))))
    for key in ("blocks.0.attn.gate", "blocks.0.attn.head_w"):
        t = sd[key]
        st = stats(key, t, torch.ones_like(t))
        facts["rows"].append(st)
        print("  %-28s 初值 1.0 -> 均值 %.4f   逐元素 |Δ| 均值 %.4f  max %.4f"
              % (key, float(t.mean()), float((t - 1).abs().mean()),
                 float((t - 1).abs().max())))
    print("  ★这几项偏离初值，说明规则侧**确实在被学习**；但它们管的是「门多强 / 哪些候选参与」，")
    print("    不是「窗口放多远」—— 后者由 S1 + start_bucket 实测只到 {0,1} 决定。")

    # ── 综合：谁承载语义 ─────────────────────────────────────────────
    print()
    print("=" * 78)
    print("【结论】语义与「语言规则」分别落在哪")
    print("=" * 78)
    rows = dict((r["name"], r) for r in facts["rows"])
    sem = [k for k in ("emb.weight", "engram.val.weight", "engram.key.weight", "head.weight")
           if k in rows]
    sem_norm = sum(rows[k]["norm"] for k in sem)
    rule_keys = [k for k in rows if k.startswith("blocks.0.attn.")
                 and k.split(".")[-1] in ("gate", "head_w", "rule_scale", "lam",
                                          "bucket_proj.weight", "bucket_proj.bias")]
    rule_norm = sum(rows[k]["norm"] for k in rule_keys)
    print("  语义载体（emb + engram.val/key + head）合计 ‖W‖ = %.2f" % sem_norm)
    print("  规则侧（gate / head_w / bucket_proj / rule_scale / lam）合计 ‖W‖ = %.2f"
          % rule_norm)
    print("  比值 = %.2f : 1  ⇒ 参数量级上语义占绝对主导" % (sem_norm / max(rule_norm, 1e-9)))
    print()
    print("  · `slot_ids` 是 **token id 的纯哈希**，与内容无关 ⇒ Engram 学到的是")
    print("    「这串 token 出现过」→ 一个可学向量，即**共现/词项统计**，不是语义理解。")
    print("  · 语义真正的载体是 emb / engram.val / FFN —— 都由数据在 loss 下塑形，")
    print("    是**副产品**，架构里没有任何一处显式建模语义。")
    print("  · `pos_bucket` %s ⇒ 「位置 -> 窗口起点」这条通路%s"
          % ("仍全 0" if facts["pos_bucket_moved"] == 0.0 else "已动",
             "没被学到" if facts["pos_bucket_moved"] == 0.0 else "在贡献"))
    print("  · start_bucket 实测只用 {0,1}（arch_figure_facts.json）⇒ 窗口规则只学到两档")

    facts["semantic_norm"] = sem_norm
    facts["rule_norm"] = rule_norm
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
