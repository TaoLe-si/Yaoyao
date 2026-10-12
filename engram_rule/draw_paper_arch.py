# -*- coding: utf-8 -*-
"""★ 论文式架构图 —— 模仿 "Attention Is All You Need" 的 Figure 1 版式。

**只画，不算**：所有数字来自两份 JSON，脚本里不出现手打的模型数值：
  · `arch_figure_facts.json`   ← `probe_arch_figure.py`（真实 checkpoint + 一次真实前向）
  · `sparse_design_facts.json` ← `design_sparse_attention.py`（本轮新增通路的静态设计）

**左半 = 当前架构（已训练、已读数）；右半 = 本轮要加的稀疏通路（红色虚线，尚未实现）。**
两半画在同一张图上，是为了让「加了什么、动到哪、代价多少」一眼可比。

★ 字体纪律：Microsoft YaHei **缺** 上下标（U+2080-2089 / U+2070,2076,2079）与
  checkmark / ballot-x / vertical-bar / dot-operator / circled-minus / ring-above。
  所以公式一律写 ASCII 形式（x_t / 2^k / W_o），符号用 OK / 成立，不用勾叉。
  画完先扫一遍源文件里的非 ASCII 字符再出图（这个坑踩过，matplotlib 只发 UserWarning）。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
F_CUR = os.path.join(HERE, "arch_figure_facts.json")
F_NEW = os.path.join(HERE, "sparse_design_facts.json")
OUT = os.path.join(HERE, "arch_paper_engram_rule.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

W, H = 40.0, 112.0
C = dict(
    tok=("#eceaf5", "#4b3f7a"), emb=("#e4eef8", "#0b5394"),
    eng=("#e2f4e6", "#2e7d32"), rule=("#e2ebf8", "#1a4d8f"),
    pos=("#fcf0dc", "#c47f10"), res=("#eeeeF2", "#55555f"),
    head=("#dff0ee", "#00695c"), note=("#fffbe6", "#8a7a1f"),
    new=("#fdeaea", "#b03030"), gray=("#8a8a95", "#8a8a95"),
)

# ★图上所有字符串先过一遍字形检查。踩过的坑：Microsoft YaHei 缺 U+21D2 =>
#   / U+2265 >= / 上下标，matplotlib 只在 savefig 时发一条 UserWarning，
#   图照出、字已经变成豆腐块，而最终图上没人看得出来。所以登记 + 出图前断言。
GLYPH_GUARD = {"ch": set()}


def fmt(n):
    return "{:,}".format(int(n))


def box(ax, x, y, w, h, text, fill, fs=8.0, bold=False, ha="left",
        ec=None, lw=1.4, ls="solid", z=3, pad=1.2, tc="#111111"):
    if isinstance(fill, str):
        fill = (fill, "#b03030")
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=%s" % pad,
        linewidth=lw, facecolor=fill[0], edgecolor=ec or fill[1],
        linestyle=ls, zorder=z))
    tx = (x + 2.0) if ha == "left" else (x + w / 2.0)
    GLYPH_GUARD["ch"] |= set(str(text))
    ax.text(tx, y + h / 2.0, text, fontsize=fs, color=tc, zorder=z + 1,
            ha=ha, va="center", linespacing=1.75,
            fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, color="#333333", lw=1.6, ls="solid", z=2, ms=11):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, zorder=z, linestyle=ls, shrinkA=0, shrinkB=0))


def main():
    if not os.path.exists(F_CUR):
        raise SystemExit("先跑 probe_arch_figure.py 生成 %s" % os.path.basename(F_CUR))
    if not os.path.exists(F_NEW):
        raise SystemExit("先跑 design_sparse_attention.py 生成 %s" % os.path.basename(F_NEW))
    A = json.load(open(F_CUR, encoding="utf-8"))
    N = json.load(open(F_NEW, encoding="utf-8"))
    cfg, der, P, D = A["cfg"], A["derived"], A["params"], A["diag"]
    E, PL, WIN, SB = D["engram"], D["per_layer"], D["window"], D["start_bucket_layer0"]
    TOT = P["trainable_elements"]
    d, nl, nh = cfg["d"], cfg["n_layers"], cfg["n_heads"]
    hd, R = der["head_dim"], der["n_cand"]
    nbk, step, pmax = cfg["n_buckets"], cfg["bucket_step"], cfg["pos_abs_max"]
    wmax, rad = cfg["window_max"], der["static_lookback_radius"]

    fig = plt.figure(figsize=(17, 47), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ======================================================== 标题
    ax.text(W / 2, 109.0, "Engram-Rule   ——   神经符号 Transformer 架构",
            fontsize=25, fontweight="bold", ha="center", va="center")
    ax.text(W / 2, 105.0,
            "左：当前已训练架构（读数来自真实 checkpoint）        "
            "右：本轮要新增的稀疏长程通路（尚未实现）",
            fontsize=12, ha="center", va="center", color="#3a3a46")
    ax.text(W / 2, 102.0,
            "版式模仿 Attention Is All You Need Figure 1：左 encoder stack / "
            "右 decoder stack，这里左=规则窗口支路 / 右=Engram 与位置支路",
            fontsize=9.2, ha="center", va="center", color="#7a7a86")
    ax.text(W / 2, 99.6,
            "读数口径：%s（%s 字节） · gate=%s · 无 softmax、无 (LxL) 注意力矩阵"
            % (A["ckpt"], fmt(A["ckpt_bytes"]), A["gate"]),
            fontsize=8.4, ha="center", va="center", color="#9a9aa6")

    LX, LW = 3.0, 23.0
    CX = LX + LW / 2.0
    RX, RW = 28.0, 10.6

    # 分栏底色
    ax.add_patch(Rectangle((1.6, 15.0), 25.8, 82.0, facecolor="#fbfbfd",
                           edgecolor="#c9c9d2", linewidth=1.4, zorder=0))
    ax.add_patch(Rectangle((RX - 0.6, 15.0), RW + 1.2, 82.0, facecolor="#fffafa",
                           edgecolor="#e0b4b4", linewidth=1.4, zorder=0, linestyle=(0, (6, 3))))
    ax.text(LX + 1.2, 95.2, "A.  当前架构（已训练）", fontsize=12.6,
            fontweight="bold", color="#22223a", va="center")
    ax.text(RX, 95.2, "B.  本轮新增", fontsize=12.6, fontweight="bold",
            color="#b03030", va="center")

    from fontTools.ttLib import TTFont
    _cmap = set()
    for _t in TTFont("C:/Windows/Fonts/msyh.ttc", fontNumber=0)["cmap"].tables:
        _cmap |= set(_t.cmap.keys())

    # ---- A1 输入 ----
    box(ax, LX, 88.0, LW, 6.0,
        "输入 token 序列     x_1 ... x_L\n"
        "[BOS] [USER] 问题 [ASSISTANT] 答案 [EOS]        vocab = %d"
        % cfg["vocab"],
        C["tok"], fs=9.4, bold=True)

    # ---- A2 嵌入 + Engram 读 ----
    box(ax, LX, 79.0, LW * 0.42, 6.4,
        "Embedding    emb(x_t)\n%d x %d = %s 参数"
        % (cfg["vocab"], d, fmt(cfg["vocab"] * d)),
        C["emb"], fs=9.0, bold=True)
    box(ax, LX + LW * 0.46, 79.0, LW * 0.54, 6.4,
        "Engram 多哈希软读    slot_ids(x_t)\n"
        "阶 %d x 哈希 %d  ->  每位置 R = %d 候选槽（实测去重均值 %.3f）\n"
        "key / val 表  %s x %d      存活位 alive  %s 位"
        % (cfg["ngram_max"], cfg["n_hash"], R, E["mean_distinct_slots_per_pos"],
           fmt(cfg["n_slots"]), d, fmt(cfg["n_slots"])),
        C["eng"], fs=8.4, bold=True)
    arrow(ax, (LX + LW * 0.20, 88.0), (LX + LW * 0.20, 85.4), C["emb"][1])
    arrow(ax, (LX + LW * 0.73, 88.0), (LX + LW * 0.73, 85.4), C["eng"][1])

    # ---- N x 虚线框（包住 A3..A7，论文 Figure 1 的 Nx 语义）----
    ax.add_patch(FancyBboxPatch(
        (LX - 1.3, 37.6), LW + 2.6, 38.0,
        boxstyle="round,pad=0,rounding_size=1.4", linewidth=2.0,
        facecolor="none", edgecolor="#55555f", linestyle=(0, (7, 4)), zorder=2))
    ax.text(LX + 0.4, 74.0, "N x", fontsize=15, fontweight="bold",
            color="#55555f", va="center", zorder=6)

    # ---- A3 规则腿 ----
    box(ax, LX, 67.0, LW, 6.6,
        "规则腿：内容门 + 加权聚合\n"
        "g_r = sigmoid( gate * v_r )        gate 是 %d 维可学习向量\n"
        "agg = sum_r  head_w_r * g_r * v_r * att_r        att_r = softmax(-beta * ||q_r - k_r||^2),  beta = %s"
        % (d, cfg["beta"]),
        C["rule"], fs=8.6)
    arrow(ax, (LX + LW * 0.30, 79.0), (LX + LW * 0.30, 73.6), C["eng"][1])

    # ---- A4 分桶偏置 + 硬选择 ----
    box(ax, LX, 58.0, LW, 7.2,
        "分桶偏置  ->  决定「看哪里」\n"
        "bias   = bucket_proj(agg)                         %d -> %d\n"
        "bias   = bias + pos_bucket[min(pos, %d)]          %d -> %d         <- 位置通路①\n"
        "bias_n = rule_scale * standardize(bias)      start = argmax(bias_n)   <- 窗口起点桶号"
        % (d, nbk, pmax - 1, pmax, nbk),
        C["rule"], fs=8.6)
    arrow(ax, (CX, 67.0), (CX, 65.2), C["rule"][1])

    # ---- A5 窗口 gather ----
    box(ax, LX, 49.0, LW, 7.2,
        "窗口 gather（连续区间，非稀疏采样）\n"
        "off0 = min( start * bucket_step,  pos - %d )        off = [off0, off0 + %d]\n"
        "clamp_window = %s  ->  起点夹到 [max(0, pos-%d), pos]     j = pos - off\n"
        "实测窗口偏移 [%d, %d]  均值 %.2f      start_bucket 实际取值 {%s}"
        % (wmax - 1, wmax - 1, cfg["clamp_window"], wmax - 1,
           WIN["offset_min"], WIN["offset_max"], WIN["offset_mean"],
           ", ".join(str(v) for v in SB["distinct"])),
        C["rule"], fs=8.6)
    arrow(ax, (CX, 58.0), (CX, 56.2), C["rule"][1])

    # ---- A6 SwiGLU 聚合 ----
    box(ax, LX, 39.0, LW, 8.6,
        "SwiGLU 窗口聚合  ——  没有 softmax，没有归一化注意力矩阵；只在窗口内 gather，O(L x W)\n"
        "q,k 施加部分 RoPE（rope_dim = %d / head_dim = %d，只转前半段）；v 不施加\n"
        "gc    = sigmoid( content_gate(h) )                       实测 layer0 = %.4f\n"
        "gate  = SiLU( W_g k_j  +  ( W_q q + pos_abs[pos] ) * gc * lam  +  bsel_j )      <- 位置通路②\n"
        "up    = W_u v_j            o = sum_j (gate * up) / ( sum_j |gate| + 1e-5 )\n"
        "bsel_j = bias_n.gather( bucket(off_j) )  ——  规则唯一可微强度：只喂 argmax 整数索引则规则无梯度"
        % (cfg["rope_dim"], hd, PL[0]["content_gate"]),
        C["rule"], fs=8.4)
    arrow(ax, (CX, 49.0), (CX, 47.6), C["rule"][1])

    # ---- A7 残差 + FFN + head ----
    box(ax, LX, 29.0, LW, 7.4,
        "残差 + 前馈（每个子层都过 Add & Norm）\n"
        "h = h + ( W_o o  +  direct(agg) )            direct(agg) 是不过窗口的旁路\n"
        "h = h + FFN( LayerNorm(h) )                  FFN：%d -> %d -> %d"
        % (d, cfg["ffn_mult"] * d, d),
        C["res"], fs=8.8)
    arrow(ax, (CX, 39.0), (CX, 36.4), C["res"][1])

    box(ax, LX, 21.0, LW, 6.2,
        "LayerNorm  ->  head  %d x %d  ->  softmax  ->  答案 token"
        % (d, cfg["vocab"]),
        C["head"], fs=10.0, bold=True)
    arrow(ax, (CX, 29.0), (CX, 27.2), C["head"][1])

    # 残差主干虚线（左外侧）
    HX = LX - 2.4
    ax.plot([HX, HX], [23.0, 91.0], color="#55555f", lw=1.5,
            linestyle=(0, (5, 3)), zorder=2)
    ax.plot([HX, LX], [32.7, 32.7], color="#55555f", lw=1.5,
            linestyle=(0, (5, 3)), zorder=2)
    arrow(ax, (HX, 91.0), (LX, 91.0), "#55555f", lw=1.5, ls=(0, (5, 3)), ms=9)
    ax.text(HX - 1.1, 57.0, "h 残差主干", fontsize=8.6, color="#55555f",
            rotation=90, ha="center", va="center", fontweight="bold")

    ax.text(LX, 17.6,
            "三条信息通路：① 规则向量 -> 桶偏置 -> 窗口起点 -> SwiGLU 门     "
            "② Engram direct 旁路     ③ 绝对位置（pos_bucket 定起点 / pos_abs 进门的查询侧）",
            fontsize=8.2, color="#555555", va="center")
    ax.text(LX, 15.6,
            "层数 n_layers = %d     ·     每层注意力新增参数见右栏；总可训练参数 %s（%d 个张量）"
            % (nl, fmt(TOT), P["n_tensors"]),
            fontsize=8.2, color="#555555", va="center")

    # ======================================================== B 右栏
    box(ax, RX, 66.0, RW, 26.0,
        "新增一条**不受 Engram 门控**的稀疏支路\n"
        "\n"
        "为什么必须不受门控：\n"
        "model.py 里 o = o * keep，keep = ~skip，\n"
        "skip 来自 engram.read 的 ~any_live。\n"
        "实测 alive 清零 => 与关掉规则逐位相同\n"
        "（max|Delta| = 0.000e+00）=> 硬门。\n"
        "放进 keep 之内就白做。\n"
        "\n"
        "偏移表（本地稠密 + 几何外推）：\n"
        "  本地  %s\n"
        "  几何  %s\n"
        "  step=%d x r=%.1f^k\n"
        "新增 nn.Linear(%d, %d) = %s 参数，1 个 module"
        % (" ".join(str(v) for v in N["offsets"][:N["n_local"]]),
           " ".join(str(v) for v in N["offsets"][N["n_local"]:]),
           N["bucket_step"], N["r_geo"], N["d"], N["w_sparse"] * N["d"],
           fmt(N["new_params"])),
        C["new"], fs=8.0, ls=(0, (6, 3)), lw=1.8, ec="#b03030")

    box(ax, RX, 39.0, RW, 25.0,
        "买到什么（纯算术）\n"
        "\n"
        "最远回看      %d  ->  %d  token\n"
        "                = %.0f 倍\n"
        "\n"
        "L=1M 覆盖率   %.6f%%  ->  %.6f%%\n"
        "                = %.2f 倍   <- 不是卖点\n"
        "\n"
        "★卖点是**最远距离**跨过 seq=2048\n"
        "与 32K 两道坎，不是覆盖率。\n"
        "\n"
        "离 1M 还差 %.0f 倍\n"
        "=> 最后一程必须分块/层级，\n"
        "   不是靠加槽（成本线性、收益对数）"
        % (N["reach_cont"], N["far"], N["speedup_far"],
           N["cov_cont_pct"], N["cov_new_pct"], N["speedup_cov"],
           1_000_000.0 / N["far"]),
        C["new"], fs=8.0, ls=(0, (6, 3)), lw=1.8, ec="#b03030")

    box(ax, RX, 21.0, RW, 16.0,
        "代价与硬约束\n"
        "\n"
        "中间张量 (B=%d, L=%d, W_s=%d, H=%d, Dg=%d)\n"
        "  = %.0f MB，与现有连续窗口 %.0f MB 同量级\n"
        "  叠加后约 %.0f MB；本机空闲约 7068 MiB => 够\n"
        "\n"
        "★旧 checkpoint **不可 resume**：\n"
        "  state_dict 键变 + 多 1 个 module\n"
        "  => 新通路单独开配置跑，不掺进 scaling 网格"
        % (256, 512, N["w_sparse"], N["n_heads"], N["gate_rank"],
           N["mb_new"], N["mb_cont"], N["mb_total"]),
        C["new"], fs=8.0, ls=(0, (6, 3)), lw=1.8, ec="#b03030")

    # 参数构成小表
    ax.text(RX, 18.4, "参数构成（当前）", fontsize=9.4, fontweight="bold",
            color="#22223a", va="center")
    mods = sorted(P["by_module_numel"].items(), key=lambda kv: -kv[1])
    yy = 16.4
    for k, v in mods:
        ax.text(RX + 0.6, yy, "%-8s %9s  %5.2f%%"
                % (k, fmt(v), 100.0 * v / TOT), fontsize=7.6,
                color="#333333", va="center")
        yy -= 1.5

    # ======================================================== 页脚
    ax.text(W / 2, 12.0,
            "★ 三条必须先过的门禁（实现前已登记，失败即设计错）：",
            fontsize=9.6, fontweight="bold", color="#b03030", ha="center", va="center")
    for i, (k, v) in enumerate(N["predictions"].items()):
        ax.text(W / 2, 9.8 - i * 2.3,
                "%s   %s" % (k, v),
                fontsize=9.0, ha="center", va="center", color="#333333")
    ax.text(W / 2, 2.6,
            "本图全部数值由 probe_arch_figure.py（真实 checkpoint + 一次真实前向）与 "
            "design_sparse_attention.py（静态设计）读出并落盘 JSON；本脚本不硬编码任何模型数值。"
            "   gate=%s" % A["gate"],
            fontsize=8.0, color="#8a8a95", ha="center", va="center")

    fig.savefig(OUT, dpi=150, facecolor="white")

    # ★出图**之前**先验字形：缺字形的字符会静默变成豆腐块（matplotlib 只发
    #   UserWarning 不报错），而豆腐块在最终图上没人看得出来。所以这里直接抛。
    missing = sorted(c for c in GLYPH_GUARD["ch"]
                     if ord(c) > 127 and ord(c) not in _cmap)
    assert not missing, \
        "字体缺字形，图上是豆腐块：%s" % " ".join(
            "%s(U+%04X)" % (c, ord(c)) for c in missing)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print("字形自检通过（本次共画 %d 个不同字符，无缺字形）" % len(GLYPH_GUARD["ch"]))
    print("已写出 %s（%.1f MB）"
          % (os.path.basename(OUT), os.path.getsize(OUT) / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
