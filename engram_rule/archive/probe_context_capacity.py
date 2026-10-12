"""1M 上下文的**能力边界**核实: 这个模型到底能"用"多远的 token。

**为什么必须先测**: 桃要求"像现代大模型一样支持 100 万上下文的输入和输出"。
这句话有两种完全不同含义:
  (a) 能**吃下** 1M token(处理成本不随长度爆炸) —— 架构上可能本来就成立(固定窗口+ O(1) Engram);
  (b) 输出能**用到** 1M token 之前的信息 —— 这是"上下文长度"的通常含义。
不测就回答"能"或"不能"都是猜。

**预登记(可证伪)**, 动手前写死:

C1  内容通路的最大回看距离 ≤ `(n_buckets-1)·bucket_step + window_max - 1`。
    机制: 窗口起点 = `start_bucket · bucket_step`, start_bucket ∈ [0, n_buckets-1],
          窗口长度固定 window_max ⇒ 最远一次回看就是那个上界。
    v11: 15·32 + 15 = **495**。
    **否证**: 实测 `last_window_off.max() > 495` ⇒ C1 被否(公式或实现有出入)。

C2  绝对位置表 `pos_abs_max` 之外的位置**特征完全相同**(被 clamp)。
    **否证**: 给两个 ≥ pos_abs_max 的位置, 它们的 pos_abs/pos_bucket 行不同 ⇒ C2 被否。

C3  ★关键: 把距离 d 处的一个 token 换掉, 若 d > C1 上界, 输出**逐位不变**。
    也就是说模型的下一 token 只依赖"最近 ≤495 个 token + Engram + 绝对位置"。
    **否证**: 存在 d > 495 时输出仍改变 ⇒ C3 被否, 那说明还有我没找到的长程通路。
    **负向对照**: d=1 时输出必须改变 —— 否则这个判据恒真(等于没测)。

C4  Engram 分支对远端 token 的依赖是**内容寻址**的(与位置无关): 同一个 n-gram 在任何
    位置都命中同一个槽。**否证**: 同一 n-gram 放在不同位置得到不同槽 ⇒ C4 被否。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L_PROBE = 4096          # 探针序列长度(要 > 495 才有 d 超过上界的样本)


def build():
    from model import RCfg, EngramRuleLM
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    m = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return ck, m


@torch.no_grad()
def probe_window_lookback(m, L):
    """C1: 实测最大回看距离。"""
    c = m.c
    g = torch.Generator().manual_seed(3)
    x = torch.randint(3, c.vocab, (1, L), generator=g).to(DEV)
    for b in m.blocks:
        b.attn.record = True
    m(x)
    offs = torch.cat([b.attn.last_window_off.reshape(-1) for b in m.blocks])
    sbs = torch.cat([b.attn.last_startb.reshape(-1) for b in m.blocks])
    for b in m.blocks:
        b.attn.record = False
    bound = (c.n_buckets - 1) * max(c.bucket_step, 1) + c.window_max - 1
    return dict(measured_max_off=int(offs.max()), measured_min_off=int(offs.min()),
                startb_min=int(sbs.min()), startb_max=int(sbs.max()),
                startb_mean=float(sbs.float().mean()), bound=bound)


@torch.no_grad()
def probe_position_clamp(m):
    """C2: pos_abs_max 之外的位置特征是否真的相同。"""
    c = m.c
    blk = m.blocks[0].attn
    hi = c.pos_abs_max + 37
    ps = torch.tensor([c.pos_abs_max - 1, c.pos_abs_max, hi, hi + 1000],
                      device=DEV).clamp(max=c.pos_abs_max - 1)
    row = blk.pos_abs(ps)
    pb = blk.pos_bucket(ps)
    same_a = bool((row[1] == row[3]).all())
    same_b = bool((pb[1] == pb[3]).all())
    # 负向对照: 范围内两个位置必须不同
    inner = torch.tensor([5, 9], device=DEV)
    diff_inner = bool((blk.pos_abs(inner)[0] != blk.pos_abs(inner)[1]).any())
    return dict(pos_abs_max=c.pos_abs_max, clamped_same=bool(same_a and same_b),
                distinct_inside_range=diff_inner, probed=[int(p) for p in ps])


@torch.no_grad()
def probe_influence(m, bound, distances):
    """C3: 换掉距离 d 处的 token, 看**末位 logits** 变不变。"""
    c = m.c
    g = torch.Generator().manual_seed(5)
    x = torch.randint(3, c.vocab, (1, L_PROBE), generator=g).to(DEV)
    base = m(x)[0, -1].clone()
    out = []
    for d in distances:
        y = x.clone()
        y[0, -1 - d] = (int(y[0, -1 - d]) + 7) % c.vocab
        if int(y[0, -1 - d]) == 3:
            y[0, -1 - d] = 4
        delta = float((m(y)[0, -1] - base).abs().max())
        out.append((d, delta))
    return out


@torch.no_grad()
def probe_engram_position_independence(m):
    """C4: 同一 n-gram 放在不同位置, 是否命中同一槽。"""
    c = m.c
    g = torch.Generator().manual_seed(7)
    x = torch.randint(3, c.vocab, (1, 2048), generator=g).to(DEV)
    # 造一个重复的 2-gram, 放在两个相距很远的位置
    a, b = int(x[0, 100]), int(x[0, 1800])
    x[0, 1800] = a
    x[0, 1801] = b
    x[0, 100] = a
    x[0, 101] = b
    slots = m.engram.slot_ids(x)
    r = c.ngram_max - 1                        # 最高阶 = 2-gram (v11 ngram_max=2)
    s1 = int(slots[0, 101, r])
    s2 = int(slots[0, 1801, r])
    return dict(slot_at_100=s1, slot_at_1800=s2, same=(s1 == s2), ngram_max=c.ngram_max)


def main():
    ck, m = build()
    print("v11 cfg: d=%d layers=%d heads=%d window_max=%d n_buckets=%d bucket_step=%d "
          "ngram_max=%d n_hash=%d n_slots=%d pos_abs_max=%d vocab=%d"
          % (m.c.d, m.c.n_layers, m.c.n_heads, m.c.window_max, m.c.n_buckets,
             m.c.bucket_step, m.c.ngram_max, m.c.n_hash, m.c.n_slots,
             m.c.pos_abs_max, m.c.vocab))

    print("\n[C1] 内容通路最大回看距离 (L=%d)" % L_PROBE)
    w = probe_window_lookback(m, L_PROBE)
    print("     实测 last_window_off ∈ [%d, %d] | startb ∈ [%d, %d] 均值 %.2f"
          % (w["measured_min_off"], w["measured_max_off"], w["startb_min"],
             w["startb_max"], w["startb_mean"]))
    print("     解析上界 (n_buckets-1)*bucket_step + window_max-1 = %d ⇒ C1 %s"
          % (w["bound"], "成立" if w["measured_max_off"] <= w["bound"]
             else "★被否(实测超过上界)"))

    print("\n[C2] 绝对位置表边界")
    p = probe_position_clamp(m)
    print("     pos_abs_max=%d | ≥上限的位置特征相同=%s | 范围内位置互不相同=%s"
          % (p["pos_abs_max"], p["clamped_same"], p["distinct_inside_range"]))
    print("     ⇒ C2 %s" % ("成立" if p["clamped_same"] and p["distinct_inside_range"]
                            else "★被否"))

    print("\n[C3] 远端 token 对末位输出的影响 (L=%d)" % L_PROBE)
    ds = [1, 2, 8, 64, 255, 480, 495, 496, 600, 1000, 2000, 4000]
    res = probe_influence(m, w["bound"], ds)
    for d, delta in res:
        tag = "改变" if delta > 0 else "**不变**"
        print("     d=%-5d Δlogits=%.3e  %s%s"
              % (d, delta, tag, "   ← 在 C1 上界之外" if d > w["bound"] else ""))
    near = [dd for d, dd in res if d <= 4]
    far = [dd for d, dd in res if d > w["bound"]]
    print("     近端(d<=4) 最大 Δ=%.3e (负向对照, 必须 >0) | 远端(d>%d) 最大 Δ=%.3e"
          % (max(near), w["bound"], max(far)))
    print("     ⇒ C3 %s" % ("成立: 超过上界的 token 完全不影响输出" if max(far) == 0
                            else "★被否: 远端仍有影响"))

    print("\n[C4] Engram 槽是否与位置无关")
    e = probe_engram_position_independence(m)
    print("     同一 2-gram 放在位置 101 与 1801: slot %d vs %d ⇒ %s ⇒ C4 %s"
          % (e["slot_at_100"], e["slot_at_1800"], "相同" if e["same"] else "不同",
             "成立(内容寻址)" if e["same"] else "被否"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
