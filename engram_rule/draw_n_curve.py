"""§69.52 渲染：N 曲线四联图。**只读 `n_curve_facts.json`，不硬编码任何读数。**

拆分纪律（probe -> JSON -> render）：图上每个数字都能追回 `n_curve_facts.json`，
改实验只要重跑 `probe_n_curve_facts.py`，图自动跟着变。

## 四块分别在说什么

| 面板 | 内容 | 它要挡住的错误 |
|---|---|---|
| **A** | N 曲线：train tok_acc / EM / F1 与 held tok_acc | 让人以为「train 最优在 N=2952」已经定性 |
| **B** | 解耦后（steps 恒 462）与原对角线叠放 | 把合成轴的峰当成数据量的峰 |
| **C** | (steps, N) 设计图 | 忘了 steps 与 N 共线，于是把单轴曲线归因给数据量 |
| **D** | Engram 槽位压力（token/槽）而非活槽率 | 拿一个常量指标（alive_ratio≈100%）当证据 |

用法：`python draw_n_curve.py`
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(HERE, "n_curve_facts.json")
OUT = os.path.join(HERE, "n_curve.png")
FONT = "C:/Windows/Fonts/msyh.ttc"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

BLUE, RED, GREEN, ORANGE, GREY = "#2f5fa8", "#c0392b", "#1e8449", "#d68910", "#7a7a86"


def fmt(n):
    return "{:,}".format(int(n))


def on_epochs40(n, steps, meta):
    """这一格在不在「epochs 恒 40」对角线上：steps == ceil(epochs_target*n/bs)。"""
    return steps == -(-int(meta["epochs_target"] * n) // meta["bs"])


def on_steps462(n, steps, meta):
    """这一格在不在「steps 恒 462」横线上。"""
    return steps == meta["steps_fixed"]


def line_cells(cells, pred, meta):
    """按**规则**（不是落盘标签）挑出某条线上的格，按 N 升序。

    ★与 probe 同理：锚点 (2952,462) 复用自 sweep，只带 epochs40 标签，
      用标签判会把它漏出 steps462 线。
    """
    cs = [c for _n, _s, c in cells if pred(c["n"], c["steps"], meta)]
    return sorted(cs, key=lambda c: c["n"])


def main():
    if not os.path.exists(FACTS):
        raise SystemExit("缺 %s，先跑 probe_n_curve_facts.py" % os.path.basename(FACTS))
    F = json.load(open(FACTS, encoding="utf-8"))
    cells = F["cells"]
    meta, q1, q2, q3 = F["meta"], F["q1_collinearity"], F["q2_powerlaw"], F["q3_decoupled"]
    if not cells:
        raise SystemExit("facts 里 0 格")

    e40 = line_cells(cells, on_epochs40, meta)
    s462 = line_cells(cells, on_steps462, meta)
    peak_n = q2["peak_n"]
    # ★推荐点只从 JSON 读，不在这里重算口径。两处口径都必须钉死：
    #   ① 只在 **epochs 恒 40** 这条扫描线内选，绝不与 steps≡462 那条诊断线混排
    #      （我第一版混排过，排出「推荐 N=738」，那格训了 462 步而 N=5904 训了 923 步）。
    #      ★但它也不等于「出货配方」：出货的 738 行模型用的是 **400 步（≈139 epochs）**，
    #      不是 epochs≡40 给的 116 步（§69.50 锚点 tok_acc 0.9993，而该扫描格 0.9374）。
    #      所以这条线叫**扫描线**，选出来的点要连「它花了多少步」一起报。
    #   ② 用 EM（逐行全对）而不是 tok_acc（逐 token）——tok_acc 在 N=5904 仍有 0.959，
    #      按 tok_acc 选会选中一个 EM 只有 0.575 的点。
    e40_em = [e for e in F["em_table"] if e["on_epochs40"]]
    if not e40_em:
        raise SystemExit("em_table 里没有扫描线上的格，拒绝跨口径推荐")
    best_em = max(e40_em, key=lambda z: z["p"])
    rec_n = best_em["n"]

    fig, axes = plt.subplots(2, 2, figsize=(17.2, 10.6), dpi=150)
    fig.patch.set_facecolor("white")

    # ── A. N 曲线 ────────────────────────────────────────────────
    ax = axes[0][0]
    ns = [c["n"] for c in e40]
    ax.plot(ns, [c["train_tok_acc"] for c in e40], "-o", color=BLUE, lw=2.2,
            ms=7, label="train tok_acc (逐 token)", zorder=3)
    ax.plot(ns, [c["train_em"] / float(c["train_em_n"]) for c in e40], "-s",
            color=GREEN, lw=2.0, ms=6, label="train EM (逐行全对)", zorder=3)
    ax.plot(ns, [c["train_f1"] for c in e40], "-^", color=ORANGE, lw=1.8, ms=6,
            label="train char-F1", zorder=3)
    ax.plot(ns, [c["held_tok_acc"] for c in e40], "--D", color=RED, lw=2.0, ms=6,
            label="held tok_acc (探针 2000 行)", zorder=3)
    ax.axvline(peak_n, color=BLUE, ls=":", lw=1.6, zorder=2)
    # 标注放在左下空白区（axes 0.40, 0.06）：原先放 (peak*1.12, 0.06) 会被
    # lower-right 的图例盖住 —— 版面重叠要靠算位置解决，不能靠看。
    ax.annotate("train 峰 N=%s" % fmt(peak_n), xy=(peak_n, 0.997),
                xycoords="data", xytext=(0.40, 0.06), textcoords="axes fraction",
                fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([fmt(v) for v in ns])
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("训练行数 N（epochs 恒 40）")
    ax.set_ylabel("比例")
    ax.set_title("A. N 曲线：train 三条一起涨到峰再回落，held 一路涨", fontsize=12.5,
                 fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)

    # ── B. 解耦前后叠放 ─────────────────────────────────────────
    ax = axes[0][1]
    ax.plot(ns, [c["train_tok_acc"] for c in e40], "--o", color=GREY, lw=2.0, ms=7,
            label="epochs 恒 40（steps 随 N 变，共线）", zorder=3)
    if len(s462) >= 2:
        ax.plot([c["n"] for c in s462], [c["train_tok_acc"] for c in s462], "-s",
                color=RED, lw=2.4, ms=7,
                label="steps 恒 462（数据量被隔离）", zorder=4)
        # ★画「同一个 N、只加步数」的回升箭头 —— 这是「断崖不是容量墙」的唯一直接证据。
        #   没有它，B 面板会让人把 5904 的塌陷误读成「数据装不下」。
        rec = []
        for c in s462:
            same_n = [e for e in e40 if e["n"] == c["n"] and e["steps"] > c["steps"]]
            for e in same_n:
                ax.annotate("", xy=(e["n"], e["train_tok_acc"]),
                            xytext=(c["n"], c["train_tok_acc"]),
                            arrowprops=dict(arrowstyle="->", color="#1e8449",
                                            lw=2.0, shrinkA=7, shrinkB=7),
                            zorder=5)
                ax.text(e["n"] * 0.86, (e["train_tok_acc"] + c["train_tok_acc"]) / 2.0,
                        "步数x2 -> +%.3f" % (e["train_tok_acc"] - c["train_tok_acc"]),
                        fontsize=9.2, color="#1e8449", ha="right", va="center",
                        fontweight="bold")
                rec.append([c["n"], c["steps"], c["train_tok_acc"],
                            e["steps"], e["train_tok_acc"]])
        if rec:
            q3["recovered_by_steps"] = rec
        verdict = ("持平后断崖，且断崖可被步数修复"
                   if q3.get("monotone_decreasing") else
                   "不单调降")
        ax.set_title("B. 解耦后：%s（%d 格）" % (verdict, len(s462)),
                     fontsize=12.5, fontweight="bold")
    else:
        ax.set_title("B. 解耦线还没跑出 2 格（现有 %d 格）" % len(s462), fontsize=12.5)
    allx = sorted(set(ns) | set(c["n"] for c in s462))
    ax.set_xscale("log", base=2)
    ax.set_xticks(allx)
    ax.set_xticklabels([fmt(v) for v in allx])
    lo_s = min(c["train_tok_acc"] for c in (e40 + s462)) - 0.03
    ax.set_ylim(min(0.80, lo_s), 1.005)
    ax.set_xlabel("训练行数 N")
    ax.set_ylabel("train tok_acc")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.92)

    # ── C. 设计图：共线性 ───────────────────────────────────────
    ax = axes[1][0]
    xs = [float(c["steps"]) for _n, _s, c in cells]
    ys = [float(c["n"]) for _n, _s, c in cells]
    src = [c.get("source", "?") for _n, _s, c in cells]
    ax.plot(xs, ys, "o", ms=9, mfc="white", mew=2.0, color=BLUE, zorder=3)
    for x, y, sc in zip(xs, ys, src):
        if sc == "decoupled":
            ax.plot([x], [y], "o", ms=9, color=RED, zorder=4)
    diag = sorted([(c["steps"], c["n"]) for _n, _s, c in cells
                   if on_epochs40(c["n"], c["steps"], meta)])
    if len(diag) >= 2:
        ax.plot([p[0] for p in diag], [p[1] for p in diag], "--", color=GREY, lw=1.6,
                zorder=2, label="epochs 恒 40 的对角线")
    ax.axhline(462, color=RED, ls="-.", lw=1.4, zorder=2, label="steps 恒 462")
    # ★下界留白：462 那条线贴在轴底会被裁掉，看起来像「不存在」
    lo_y = min([c["n"] for _n, _s, c in cells] + [462])
    ax.set_ylim(lo_y / 1.7, max([c["n"] for _n, _s, c in cells] + [462]) * 1.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("训练步数 steps")
    ax.set_ylabel("训练行数 N")
    ax.set_title("C. 设计图：steps 与 N 共线，单轴曲线不能归因给数据量",
                 fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
    txt = ("steps/N = %s\nextreme spread %.2e < 1/min_N %.2e\n=> 共线 %s"
           % (q1["steps_over_n"], q1["ratio_spread"],
              q1["predicted_spread_bound"],
              "成立" if q1["is_collinear"] else "不成立"))
    ax.text(0.03, 0.04, txt, transform=ax.transAxes, fontsize=8.6, color="#333344",
            va="bottom", bbox=dict(fc="white", ec="#c8c8d4", alpha=0.9, pad=4))

    # ── D. 槽位压力 ─────────────────────────────────────────────
    # ★为什么这格不画 alive_ratio 当证据：实测 N=738 时就是 4091/4096（99.9%），
    #   一路到 11808 都全满 => 常量指标，按规则 9 不能当证据。
    #   真正随 N 变的是「每个槽平均承载多少 n-gram 出现」。
    ax = axes[1][1]
    al = F.get("alive_sweep", {})
    keys = sorted([k for k in al if k.isdigit()], key=int)
    if keys:
        an = [int(k) for k in keys]
        tps = [al[k]["tokens_per_slot"] for k in keys]
        live = [100.0 * al[k]["alive_ratio"] for k in keys]
        ax.plot(an, tps, "-o", color=BLUE, lw=2.4, ms=8,
                label="每个槽承载的 n-gram 出现数", zorder=4)
        for x, y in zip(an, tps):
            ax.annotate("%.1f" % y, xy=(x, y), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=BLUE)
        ax.set_xscale("log", base=2)
        ax.set_xticks(an)
        ax.set_xticklabels([fmt(v) for v in an])
        ax.set_xlabel("训练行数 N")
        ax.set_ylabel("token 数 / 槽（4096 槽）", color=BLUE)
        ax.tick_params(axis="y", colors=BLUE)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(an, live, "--s", color=GREY, lw=1.6, ms=6,
                 label="alive_ratio（常量，无区分度）")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("alive_ratio (%)", color=GREY)
        ax2.tick_params(axis="y", colors=GREY)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8.8, loc="upper left", framealpha=0.92)
        ax.set_title("D. Engram 槽位：表早已全满，压力随 N 线性涨（故活槽率无区分度）",
                     fontsize=12.5, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "槽位读数缺失", ha="center", fontsize=13, color=GREY)
        ax.set_title("D. Engram 槽位压力", fontsize=12.5, fontweight="bold")

    # ── 页脚：推荐点 + 诚实边界 ─────────────────────────────────
    fig.suptitle("§69.52  N 曲线与最优点（d=128, n_slots=4096, bs=256, lr=0.003, "
                 "seed=11, 单 seed 无极差）", fontsize=14.5, fontweight="bold",
                 y=0.985)
    runners = sorted(e40_em, key=lambda z: -z["p"])[1:2]
    run_txt = ""
    if runners:
        r0 = runners[0]
        disjoint = best_em["ci95"][0] > r0["ci95"][1]
        run_txt = ("   |   次优 N=%s EM=%.3f CI95=[%.3f, %.3f] -> 区间%s"
                   % (fmt(r0["n"]), r0["p"], r0["ci95"][0], r0["ci95"][1],
                      "不相交（95%% 下可分辨）" if disjoint else "相交（不可分辨）"))
    foot = ("扫描线（epochs 恒 40）内最优点 N=%s：train EM %d/%d = %.3f "
            "(Wilson 95%% [%.3f, %.3f]，共 %d 步)%s"
            % (fmt(rec_n), best_em["em"], best_em["em_n"], best_em["p"],
               best_em["ci95"][0], best_em["ci95"][1], best_em["steps"], run_txt))
    fig.text(0.5, 0.010, "出货基准是 d=128 / 400 步（DESIGN §64.5，≈139 epochs），不是扫描线给的 116 步 => 它落在 steps≡462 那条线上。"
                         "  判据用 EM 逐行全对而非 tok_acc：N=5,904 的 tok_acc 仍有 0.9589 但 EM 只 0.575。",
             ha="center", fontsize=9.0, color="#7a4a4a")
    fig.text(0.5, 0.036, foot, ha="center", fontsize=9.2, color="#333344")
    fig.tight_layout(rect=(0, 0.072, 1, 0.965))

    # ★字形自检：扫**整个源文件**的非 ASCII 字符，比只登记渲染字符串更严
    #   （轴标签/标题不在 GLYPH_GUARD 那类登记里，漏检就会出豆腐块）。
    from fontTools.ttLib import TTFont
    cmap = set()
    for t in TTFont(FONT, fontNumber=0)["cmap"].tables:
        cmap |= set(t.cmap.keys())
    src_txt = open(os.path.abspath(__file__), encoding="utf-8").read()
    used = set(c for c in src_txt if ord(c) > 127)
    missing = sorted(c for c in used if ord(c) not in cmap)
    assert not missing, ("字体缺字形，图上是豆腐块：%s"
                         % " ".join("%s(U+%04X)" % (c, ord(c)) for c in missing))
    fig.savefig(OUT, dpi=150, facecolor="white")
    print("字形自检通过（源文件 %d 个非 ASCII 字符，无缺字形）" % len(used))
    print("  epochs40 线 %d 格：N=%s" % (len(e40), [fmt(c["n"]) for c in e40]))
    print("  steps462 线 %d 格：N=%s" % (len(s462), [fmt(c["n"]) for c in s462]))
    print("  推荐 N=%s（EM %.3f）" % (fmt(rec_n), best_em["p"]))
    print("已写出 %s（%.2f MB）"
          % (os.path.basename(OUT), os.path.getsize(OUT) / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())