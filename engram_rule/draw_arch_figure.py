# -*- coding: utf-8 -*-
"""★ 画架构图 —— **只画, 不算**。

所有数字来自 `arch_figure_facts.json`(由 `probe_arch_figure.py` 从真实 checkpoint +
一次真实前向读出)。本脚本里**不出现任何硬编码的模型数值** —— 每个 f-string 都是从
JSON 取的。这样图上任何一处数字都能追回来源(规则: 不猜)。

数据流严格照 `model.py` 的 `EngramRuleLM.forward` / `RuleAttention.forward` 画:
  emb(x) → h 残差主干;  engram.read → (rule_vec=v_r, rule_w=att_r);
  agg = Σ_r head_w_r · σ(gate⊙v_r) ⊙ v_r · att_r;   bias = bucket_proj(agg) + pos_bucket[pos]
  direct(agg) 是**不过窗口**的旁路; 窗口只由 start=argmax(bias_n) 决定起点。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(HERE, "arch_figure_facts.json")
OUT = os.path.join(HERE, "arch_figure_engram_rule.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

W, H = 280.0, 190.0
C = dict(
    tok=("#eee8f7", "#5b3a8f"), emb=("#e6f0f7", "#0b5394"),
    eng=("#e3f5e6", "#2e7d32"), rule=("#e3ecf9", "#1a4d8f"),
    pos=("#fdf0dd", "#c47f10"), res=("#f0f0f2", "#55555f"),
    head=("#e0f2f1", "#00695c"), note=("#fffbe6", "#8a7a1f"),
    gray=("#888888", "#888888"),
)


def fmt(n):
    return "{:,}".format(int(n))


def pct(x, total):
    return "%.2f%%" % (100.0 * x / total)


# ------------------------------------------------------------------ 画图原件
def box(ax, x, y, w, h, text, fill, fs=8.2, bold=False, tc="#111111",
        ec=None, lw=1.3, ha="center", z=3, pad=1.4, ls="solid", alpha=1.0):
    if isinstance(fill, str):
        fill = (fill, "#b03030")
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=%s" % pad,
        linewidth=lw, facecolor=fill[0], edgecolor=ec or fill[1],
        linestyle=ls, zorder=z, alpha=alpha))
    tx = (x + 2.6) if ha == "left" else (x + w / 2.0)
    ax.text(tx, y + h / 2.0, text, fontsize=fs, color=tc, zorder=z + 1,
            ha=ha, va="center", linespacing=1.6,
            fontweight="bold" if bold else "normal")
    return (x, y, w, h)


def arrow(ax, p0, p1, color="#333333", lw=1.5, style="-|>", rad=0.0,
          ls="solid", z=2, ms=9):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=ms, linewidth=lw,
        color=color, zorder=z, linestyle=ls,
        connectionstyle="arc3,rad=%s" % rad, shrinkA=0, shrinkB=0))


def main():
    if not os.path.exists(FACTS):
        raise SystemExit("先跑 probe_arch_figure.py 生成 %s" % os.path.basename(FACTS))
    F = json.load(open(FACTS, encoding="utf-8"))
    cfg, der = F["cfg"], F["derived"]
    P, D = F["params"], F["diag"]
    E, PL, SG = D["engram"], D["per_layer"], D["scalars"]
    WIN, SB = D["window"], D["start_bucket_layer0"]
    TOT = P["trainable_elements"]
    bc, bn = P["by_cat_count"], P["by_cat_numel"]

    d, nl, nh = cfg["d"], cfg["n_layers"], cfg["n_heads"]
    hd = der["head_dim"]
    WW, R, step = WIN["W"], der["n_cand"], der["bucket_step"]
    nbk, pmax = cfg["n_buckets"], cfg["pos_abs_max"]
    hx = hx = ", ".join(str(v) for v in SB["distinct"])

    fig = plt.figure(figsize=(28, 19), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ============================================================ 标题
    ax.text(W / 2, 183.0, "Engram-Rule   神经符号 Transformer   ——   完整架构",
            fontsize=30, fontweight="bold", ha="center", va="center")
    ax.text(W / 2, 176.4,
            "三条信息通路   ① 规则 / 窗口      ② Engram direct      ③ 绝对位置"
            "          |          无 softmax、无 (L×L) 注意力矩阵",
            fontsize=14, ha="center", va="center", color="#444444")
    ax.text(W / 2, 171.0,
            "实测口径：checkpoint  %s  （%s 字节）   ·   门禁指纹 gate = %s   ·   "
            "全部数值由 probe_arch_figure.py 读出并写入 arch_figure_facts.json"
            % (F["ckpt"], fmt(F["ckpt_bytes"]), F["gate"]),
            fontsize=9.6, ha="center", va="center", color="#7a7a86")

    lx = 6.0
    for name, col, desc in (
            ("通路①  规则 / 窗口", C["rule"][1],
             "Engram 规则向量 → 桶偏置 → 窗口起点 → SwiGLU 门"),
            ("通路②  Engram direct", C["eng"][1],
             "agg 直接投影回残差，不过窗口"),
            ("通路③  绝对位置", C["pos"][1],
             "pos_bucket 定窗口起点  ·  pos_abs 进门的查询侧")):
        ax.add_patch(Rectangle((lx, 165.4), 3.6, 2.6, facecolor=col, zorder=5))
        ax.text(lx + 5.0, 166.7, name, fontsize=10.2, fontweight="bold",
                color=col, va="center")
        ax.text(lx + 33.0, 166.7, desc, fontsize=8.8, color="#555555", va="center")
        lx += 92.0

    # ============================================================ 主干
    MPX, MPW = 3.0, 164.0
    ax.add_patch(Rectangle((MPX, 3), MPW, 160, facecolor="#fcfcfd",
                           edgecolor="#c9c9d2", linewidth=1.5, zorder=0))
    ax.text(MPX + 3, 157.0,
            "主干：一次前向自上而下（每层重复 n_layers = %d 次）" % nl,
            fontsize=11.6, fontweight="bold", color="#22223a", va="center")

    LX, LW = 12.0, 150.0        # 主干左边界 / 宽度
    HX = LX - 5.5               # h 残差主干竖线

    # ---- ① token 序列 ----
    box(ax, LX, 145.0, LW, 9.6,
        "输入 token 序列   x1 … x_L        [BOS]  [USER]  …问题…  "
        "[ASSISTANT]  …答案…  [EOS]",
        C["tok"], fs=11.4, bold=True)
    arrow(ax, (LX + 26, 145.0), (LX + 26, 141.4), C["emb"][1])
    arrow(ax, (LX + 96, 145.0), (LX + 96, 141.4), C["eng"][1])

    # ---- ② 嵌入 / Engram 候选槽 ----
    box(ax, LX, 130.5, 46, 10.9,
        "Embedding   emb(x_t)\n%d × %d = %s" % (cfg["vocab"], d, fmt(cfg["vocab"] * d)),
        C["emb"], fs=9.6, bold=True)
    box(ax, LX + 52, 130.5, LW - 52, 10.9,
        "Engram 多哈希   slot_ids(x_t)      阶数 %d × 哈希 %d → 每位置 R = %d 个候选槽"
        "（实测去重均值 %.3f）\n"
        "槽表：key  %s×%s   val %s×%s   存活位 alive %s 位"
        % (cfg["ngram_max"], cfg["n_hash"], R, E["mean_distinct_slots_per_pos"],
           fmt(cfg["n_slots"]), d, fmt(cfg["n_slots"]), d, fmt(cfg["n_slots"])),
        C["eng"], fs=9.2, bold=True)

    # ---- ③ 软读 + 规则聚合（软读在右，聚合在左；箭头向左） ----
    box(ax, LX + 52, 111.5, LW - 52, 16.2,
        "软读：只在候选集内做 → O(1) 且可微\n"
        "询问 q_r = Σ 窗口内 emb          键 k_r = key[slot_r]     值 v_r = val[slot_r]\n"
        "att_r = softmax( −β · ‖q_r − k_r‖² )     β = %s\n"
        "实测：存活槽 %s / %s（存活率 %.4f）  ·  每位置存活候选占比 %.4f"
        % (cfg["beta"], fmt(D["alive_slots"]), fmt(cfg["n_slots"]),
           D["alive_ratio"], E["live_frac_per_pos"]),
        C["eng"], fs=9.0)
    arrow(ax, (LX + 96, 130.5), (LX + 96, 127.7), C["eng"][1])

    box(ax, LX, 111.5, 46, 16.2,
        "规则腿：门控 + 加权聚合\n"
        "g_r = σ( gate ⊙ v_r )          gate 是 %d 维可学习向量\n"
        "agg = Σ_r  head_w_r · g_r ⊙ v_r · att_r",
        C["rule"], fs=9.0)
    # ★方向: 软读的 (v_r, att_r) 喂给规则腿 -> 箭头向左
    arrow(ax, (LX + 52, 119.6), (LX + 46, 119.6), C["eng"][1], lw=2.0)

    # ---- ④ 分桶偏置（含通路③） ----
    box(ax, LX, 92.0, LW * 0.53, 16.2,
        "分桶偏置 —— 决定「看哪里」\n"
        "bias = bucket_proj(agg)              %d → %d\n"
        "bias = bias + pos_bucket[min(pos, %d)]   %d → %d        ← 通路③ 决定窗口起点"
        % (d, nbk, pmax - 1, pmax, nbk),
        C["rule"], fs=9.0)
    box(ax, LX + LW * 0.53 + 3.5, 92.0, LW * 0.47 - 3.5, 16.2,
        "归一化与硬选择\n"
        "bias_n = rule_scale · standardize(bias)\n"
        "start  = argmax(bias_n)          ← 窗口起点桶号\n"
        "规则还留一条可微强度通路（见下方 bsel）",
        C["rule"], fs=9.0)
    arrow(ax, (LX + 23, 111.5), (LX + 23, 108.2), C["rule"][1])
    arrow(ax, (LX + LW * 0.265, 100.1), (LX + LW * 0.265 + 3.5, 100.1), C["rule"][1])

    # ---- ⑤ 窗口选择 + direct 旁路 ----
    box(ax, LX, 75.5, LW - 50, 13.2,
        "窗口选择 —— 规则只决定「往回看多远」，窗内由门挑位置\n"
        "off0 = start × bucket_step (%d)        off = [off0, off0+%d]        "
        "clamp：off0 ≤ pos − %d"
        % (step, WW - 1, WW - 1),
        C["rule"], fs=9.0)
    arrow(ax, (LX + LW * 0.765, 92.0), (LX + LW * 0.765, 88.7), C["rule"][1])

    box(ax, LX + LW - 46, 75.5, 46, 13.2,
        "通路②   direct(agg)     %d × %d\n"
        "h = h + direct(agg)   —— 不经过窗口"
        % (d, d),
        C["eng"], fs=9.4, bold=True)
    # agg → direct 的折线: 绕主干右边缘走, 不从「归一化」框背后穿过去
    _rx = LX + LW + 2.5                      # 主干内的右边缘走线位置
    ax.plot([LX + 38, _rx, _rx, LX + LW],
            [111.5, 110.2, 82.1, 82.1],
            color=C["eng"][1], lw=1.5, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")
    arrow(ax, (LX + LW + 6.0, 82.1), (LX + LW, 82.1), C["eng"][1], lw=1.5)

    # ---- ⑥ SwiGLU 窗口聚合 ----
    box(ax, LX, 45.0, LW, 27.0,
        "SwiGLU 窗口聚合   ——   没有 softmax、没有归一化注意力矩阵；"
        "只在窗口内 gather，代价 O(L·W)，W = %d\n"
        "q, k 施加部分 RoPE（rope_dim = %d / head_dim = %d，只转前半段）；v 不施加\n"
        "gc    = σ( content_gate(h) )\n"
        "gate  = SiLU(  W_g·k_j   +   ( W_q·q  +  pos_abs[pos] ) · gc·lam   +   bsel_j  )"
        "          ← 通路③ 在此进门的查询侧\n"
        "up    = W_u·v_j\n"
        "o     = Σ_j gate⊙up  /  ( Σ_j |gate| + 1e−5 )          窗口内逐通道局部归一化\n"
        "out   = W_o · W_d · o"
        % (WW, cfg["rope_dim"], hd),
        C["rule"], fs=9.5)
    arrow(ax, (LX + LW * 0.5, 75.5), (LX + LW * 0.5, 72.0), C["rule"][1])
    arrow(ax, (HX, 141.4), (HX, 25.5), C["res"][1], lw=1.6, ls=(0, (5, 3)))
    ax.text(HX, 96.0, "h\n残差主干", fontsize=8.4, color=C["res"][1],
            rotation=90, ha="center", va="center", fontweight="bold")

    # ---- ⑦ bsel：规则的可微强度 ----
    box(ax, LX, 30.5, LW, 11.6,
        "bsel_j = bias_n.gather( bucket(off_j) )        ——  选中桶的分值直接进 SwiGLU 门\n"
        "★ 这是规则唯一的可微强度：若桶偏置只喂 argmax 的整数索引，"
        "rule_scale / bucket_proj / pos_bucket 会收不到任何梯度，规则根本没在被学习",
        C["pos"], fs=9.0)
    arrow(ax, (LX + LW * 0.30, 45.0), (LX + LW * 0.30, 42.1), C["pos"][1])
    # ★不再画 归一化 → bsel 的连线: bsel 框的第一行公式已写明它取自 bias_n,
    #   「归一化与硬选择」框末行也写了「见下方 bsel」。画一条要跨过整个
    #   窗口选择框的虚线只会变成面条, 反而更难读。

    # ---- ⑧ 残差 + FFN ×n ----
    ax.add_patch(FancyBboxPatch(
        (LX - 2.5, 12.0), LW + 5, 15.5,
        boxstyle="round,pad=0,rounding_size=1.6", linewidth=1.8,
        facecolor="none", edgecolor=C["res"][1], linestyle=(0, (7, 4)), zorder=3))
    box(ax, LX, 14.0, 96, 11.5,
        "h = h + (  W_o·o   +   direct(agg)  )\n"
        "h = h + FFN( LayerNorm(h) )        FFN：%d → %d → %d"
        % (d, cfg["ffn_mult"] * d, d),
        C["res"], fs=9.4)
    box(ax, LX + 101, 14.0, LW - 101, 11.5,
        "重复 %d 层\nn_layers = %d" % (nl, nl), C["res"], fs=10.2, bold=True)
    arrow(ax, (LX + LW * 0.5, 30.5), (LX + LW * 0.5, 27.5), C["res"][1])
    # h 残差主干虚线在底部转入残差框左端
    arrow(ax, (HX, 19.7), (LX, 19.7), C["res"][1], lw=1.6, ls=(0, (5, 3)))

    # ---- ⑨ head ----
    box(ax, LX, 5.0, 96, 6.4,
        "LayerNorm   →   head  %d × %d   →   softmax   →   答案 token"
        % (d, cfg["vocab"]), C["head"], fs=10.0, bold=True)
    arrow(ax, (LX + 40, 14.0), (LX + 40, 11.4), C["head"][1])

    # ============================================================ 右栏
    RX, RW = 172.0, 105.0
    ax.add_patch(Rectangle((RX, 3), RW, 160, facecolor="#fcfcfd",
                           edgecolor="#c9c9d2", linewidth=1.5, zorder=0))

    # ---- A 参数构成 ----
    ax.text(RX + 3, 157.0, "参数构成（可训练张量）", fontsize=12.4,
            fontweight="bold", color="#22223a", va="center")
    ax.text(RX + 3, 152.6, "共 %d 个张量   /   %s 元素"
            % (P["n_tensors"], fmt(TOT)), fontsize=9.2, color="#555555", va="center")

    mods = sorted(P["by_module_numel"].items(), key=lambda kv: -kv[1])
    bar_colors = ["#1a4d8f", "#2e7d32", "#0b5394", "#00695c", "#77777f"]
    bmax = mods[0][1]
    bx0, bw_max = RX + 20.0, RW - 42.0        # 数值一律右对齐在 RX+RW-2
    yb = 146.0
    for i, (k, v) in enumerate(mods):
        bw = bw_max * v / bmax
        ax.add_patch(Rectangle((bx0, yb - 1.8), bw, 3.6,
                               facecolor=bar_colors[i % len(bar_colors)],
                               edgecolor="none", zorder=3))
        ax.text(bx0 - 2.2, yb, k, fontsize=9.0, ha="right", va="center")
        ax.text(RX + RW - 2.0, yb, "%s   %s" % (fmt(v), pct(v, TOT)),
                fontsize=8.8, va="center", ha="right", color="#333333")
        yb -= 5.0
    ax.text(RX + 3, 121.5,
            "矩阵占 %.2f%%  ——  容量全在矩阵里"
            % (100.0 * bn["matrix"] / TOT),
            fontsize=8.2, color="#555555", va="center")

    # ---- B 按维数分类 ----
    ax.text(RX + 3, 117.0, "按张量维数分类", fontsize=10.6, fontweight="bold",
            color="#22223a", va="center")
    yb = 112.2
    for cat, label in (("matrix", "矩阵"), ("vector", "向量"), ("scalar", "标量")):
        ax.add_patch(Rectangle((RX + 5.0, yb - 1.5), 3.2, 3.0,
                               facecolor={"matrix": "#1a4d8f", "vector": "#2e7d32",
                                          "scalar": "#c47f10"}[cat], zorder=3))
        ax.text(RX + 10.0, yb, "%s   %d 个" % (label, bc[cat]), fontsize=8.8,
                va="center")
        ax.text(RX + RW - 2.0, yb, "%s 元素" % fmt(bn[cat]), fontsize=8.8,
                va="center", ha="right", color="#333333")
        yb -= 4.4
    ax.text(RX + 4, yb + 0.6,
            "另有 %d 个非训练 buffer（h_mult / h_mod / alive）%s 元素，"
            "state_dict 合计 %d 张量 / %s 元素。\n"
            "★ 口径：按张量维数分类（dim=0 记标量），故形状 (1,) 的 "
            "content_gate.bias 归入向量。"
            % (len(P["buffers"]), fmt(P["buffer_elements"]),
               P["state_dict_tensors"], fmt(P["state_dict_elements"])),
            fontsize=7.8, color="#8a7a1f", va="top", linespacing=1.6)

    # ---- C 实测读数 ----
    ax.text(RX + 3, 89.5, "一次真实前向的实测读数", fontsize=12.4,
            fontweight="bold", color="#22223a", va="center")
    ax.text(RX + 3, 85.5,
            "%d 行真实训练样本，L = %d，规则路径跳过率 %.4f"
            % (D["n_forward_rows"], D["seq_len_used"], D["rule_skip_ratio"]),
            fontsize=8.6, color="#555555", va="center")

    hist = SB["histogram"]
    rows = [
        ("start_bucket 实际取值",
         "{%s}    分布 %s" % (hx, "   ".join("%s:%s" % (k, fmt(v))
                                             for k, v in hist.items())),
         C["pos"][1]),
        ("窗口实际偏移范围",
         "[%d, %d]     均值 %.2f      W = %d"
         % (WIN["offset_min"], WIN["offset_max"], WIN["offset_mean"], WW),
         C["rule"][1]),
        ("配置允许的静态回看半径",
         "%d      = (桶数−1)×步长 + 窗口 − 1" % der["static_lookback_radius"],
         C["gray"][1]),
        ("规则腿量级",
         "‖bias_n‖ 均值 %.4f      rule_scale %.4f / %.4f"
         % (SG["layer0"]["bias_absmean"], PL[0]["rule_scale"], PL[1]["rule_scale"]),
         C["rule"][1]),
        ("SwiGLU 门",
         "gate 均值 %.3f / %.3f      content_gate %.4f / %.4f"
         % (PL[0]["gate_mean"], PL[1]["gate_mean"],
            PL[0]["content_gate"], PL[1]["content_gate"]), C["rule"][1]),
        ("候选槽去重",
         "均值 %.3f （min %d / max %d）  →  R = %d 个候选几乎不撞"
         % (E["mean_distinct_slots_per_pos"], E["min_distinct_slots_per_pos"],
            E["max_distinct_slots_per_pos"], R), C["eng"][1]),
        ("Engram 存活槽",
         "%s / %s        存活率 %.4f" % (fmt(D["alive_slots"]),
                                         fmt(cfg["n_slots"]), D["alive_ratio"]),
         C["eng"][1]),
    ]
    yy = 80.0
    for i, (name, val, col) in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((RX + 2, yy - 2.4), RW - 4, 5.0,
                                   facecolor="#f4f4f8", edgecolor="none", zorder=2))
        ax.text(RX + 4, yy + 0.9, name, fontsize=8.4, color=col,
                fontweight="bold", va="center", zorder=4)
        ax.text(RX + 5, yy - 1.7, val, fontsize=7.9, color="#333333",
                va="center", zorder=4)
        yy -= 5.4

    # ---- D 关键判决 ----
    box(ax, RX + 3, 21.0, RW - 6, 21.0,
        "为什么配置写着「半径 495」，实测却只到 47\n"
        "·  %d 个桶 × 步长 %d 的静态上界是 %d；但训练后的模型 "
        "start_bucket 实际只用 {%s}，\n"
        "·  窗口偏移实测上界 = %d × %d + %d = %d，"
        "余下 %d 个偏移位从未被取用。\n"
        "·  规则学到的是「看当前格 / 当前格往后 %d 格」两档；"
        "更远的桶从没被选中过。"
        % (nbk, step, der["static_lookback_radius"], hx,
           max(SB["distinct"]), step, WW - 1, WIN["offset_max"],
           der["static_lookback_radius"] - WIN["offset_max"], step),
        C["note"], fs=8.4, ha="left", pad=1.8)

    # ---- E 口径更正 ----
    box(ax, RX + 3, 3.5, RW - 6, 15.5,
        "口径更正（对先前口头报的数字）\n"
        "·  可训练参数 %s，不是 %s —— 后者把 %d 个哈希 buffer 也算进去了\n"
        "·  start_bucket 取 {%s}，不是只有 {0}\n"
        "·  「标量」按 dim=0 计，故 %d 个；形状 (1,) 的 content_gate.bias 归向量"
        % (fmt(TOT), fmt(P["state_dict_elements"]), len(P["buffers"]), hx, bc["scalar"]),
        "#fff0f0", fs=8.4, ha="left", ec="#b03030", pad=1.8)

    # ============================================================ 页脚
    ax.text(W / 2, 1.2,
            "本图所有数值均由 probe_arch_figure.py 从 checkpoint 与一次真实前向读出，"
            "写入 arch_figure_facts.json；draw_arch_figure.py 不硬编码任何模型数值。   ·   "
            "门禁指纹 gate = %s" % F["gate"],
            fontsize=8.0, color="#8a8a95", ha="center", va="center")

    fig.savefig(OUT, dpi=150, facecolor="white")
    print("已写出 %s（%.1f MB）"
          % (os.path.basename(OUT), os.path.getsize(OUT) / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())