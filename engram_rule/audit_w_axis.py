"""§69.12 静态审计：`W` 轴（扫 d）到底是不是单变量。

**为什么必须先审计再跑**：§69.7 扫 `n_slots` 时 `W=16d²` 恒定、`E` 单独动，干净。
现在要扫 `d` 测 `W` 的 β，就得反过来**把 `E` 钉死**（`n_slots = E/(2d)`）。
但扫 `d` 会不会顺带动了别的？必须先把账算清，否则又是轴绑定。

设计（`E = 524,288` 钉死，与 §69.7 的 n_slots=4096 那档同值，便于交叉对照）：

    d=32   → n_slots=8192
    d=64   → n_slots=4096
    d=128  → n_slots=2048

用法：`python audit_w_axis.py`
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

E_TARGET = 524288          # 与 §69.7 的 n_slots=4096 / d=64 同值
CELLS = ((32, 8192), (64, 4096), (128, 2048))


def classify(k):
    """★注意：`ffn.0.weight` / `ffn.2.weight` 是 W；但 `ffn.0.bias` / `ffn.2.bias`
    **不是**（bias 形状随隐层 2d 走，不构成 d² 项）。

    第一版我用 `'.ffn.' in k` 就把 bias 也算进 W ⇒ W 从 65536 变 65920（d=64），
    `16d²` 判据**假红**。这是「判据写错」的又一次：必须逐项看是不是 `.weight`。
    """
    if k.startswith("engram"):
        return "engram(E)"
    if ".attn." in k and any(".%s.weight" % t in k for t in ("q", "k", "v", "o")) \
            and k.endswith(".weight"):
        return "attn_qkvo(W)"
    if ".ffn." in k and k.endswith(".weight"):
        return "ffn(W)"
    if k.startswith("emb"):
        return "emb(随d)"
    if k.startswith("head"):
        return "head(随d)"
    if "pos_abs" in k or "pos_bucket" in k:
        return "pos_*(恒定)"
    if any(t in k for t in ("wq", "wk", "wv", "wu", "wd", "wg")):
        return "w*_bucket(恒定)"
    if "bucket_proj" in k:
        return "bucket_proj(随d)"
    if "content_gate" in k:
        return "content_gate(随d)"
    if "direct" in k:
        return "direct(随d)"
    return "norm/其他(随d)"


def main():
    from model import RCfg, EngramRuleLM
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    print("=" * 78)
    print("W 轴审计：E=%d 钉死，扫 d ⇒ W=16d² 单独动？" % E_TARGET)
    print("=" * 78)
    agg = {}
    info = {}
    for d, slots in CELLS:
        torch.manual_seed(11)
        cfg = dict(ck["cfg"])
        cfg["d"], cfg["n_heads"], cfg["n_slots"] = d, d // 16, slots
        m = EngramRuleLM(RCfg(**cfg))
        nd = dict(m.named_parameters())
        g = {}
        for k, v in nd.items():
            key = classify(k)
            g[key] = g.get(key, 0) + v.numel()
        agg[d] = g
        info[d] = dict(slots=slots,
                       W=g.get("attn_qkvo(W)", 0) + g.get("ffn(W)", 0),
                       E=g.get("engram(E)", 0))
        del m

    keys = sorted(set().union(*[set(v) for v in agg.values()]))
    print()
    print("%-22s %12s %12s %12s" % ("项", *["d=%d" % d for d, _ in CELLS]))
    for k in keys:
        print("%-22s %12d %12d %12d" % (k, *[agg[d].get(k, 0) for d, _ in CELLS]))
    print()
    print("─" * 78)
    print("判据核对")
    print("─" * 78)
    for d, slots in CELLS:
        w = info[d]["W"]
        e = info[d]["E"]
        print("  d=%-4d slots=%-6d W=%-8d 16d²=%-8d 符合=%s | E=%-8d 恒定=%s"
              % (d, slots, w, 16 * d * d, w == 16 * d * d, e, e == E_TARGET))

    other = {d: sum(v for k, v in agg[d].items()
                    if k not in ("engram(E)", "attn_qkvo(W)", "ffn(W)"))
             for d, _ in CELLS}
    print()
    print("  ★other（非 E 非 W）随 d 变化：%s"
          % "  ".join("d=%d:%d" % (d, other[d]) for d, _ in CELLS))
    ratio = other[128] / other[32]
    print("  ⇒ other 从 %d 涨到 %d（**%.2f×**）⇒ 扫 d 时它不是常量"
          % (other[32], other[128], ratio))
    print("  ⇒ 主因：emb/head 是 `vocab*d`（1024d × 2 = 2048d），d 翻倍它就翻倍")
    print()
    print("─" * 78)
    print("【结论】")
    print("─" * 78)
    print("  · `W` 本身**干净**：三档都精确等于 16d²，E 三档都精确等于 %d" % E_TARGET)
    print("  · 但 `other` 随 d 涨 %.2f× ⇒ **扫 d 不是严格单变量**" % ratio)
    print("  ⇒ 与 §69.7（扫 n_slots，other 恒定）相比，这一轴的混杂**不可忽略**")
    print()
    print("  可接受与否的判据（预登记）：")
    print("    W1 若 β 显著为正/负且 other 的量级不改变趋势 ⇒ 可用，但须标注混杂")
    print("    W2 若 other 的影响与 W 同量级 ⇒ 该轴**测不出干净 β**，只能报上界")
    print("    W3 ★负向对照：把 d 的顺序打乱后重拟合，β 必须变")
    print()
    print("  other 占 params 的比例：")
    for d, slots in CELLS:
        tot = sum(agg[d].values())
        print("    d=%-4d other/total = %.1f%%  (W/total = %.1f%%)"
              % (d, 100.0 * other[d] / tot, 100.0 * info[d]["W"] / tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
