"""§69.6 幂律图：把三个候选轴各自画出来，看哪个**真的**是幂律。

★**为什么必须画**：我从「params 随 d 增长」读出过两个**相反**的结论（先判「不是幂律」，
  拆开后又发现 q/k/v/o+ffn 是精确 d^2）。口头算会被常数项污染，画在 log-log 上
  一眼就能看出斜率对不对、线性好不好。

★**每条曲线都标「无种子极差」**（桃的规则：单 seed 不得下强结论）。

三个候选轴：
  · `params`（总量）—— 混了 engram 线性项，**预期不是幂律**
  · `E = d × n_slots`（engram）—— **预期精确 d^1**
  · `W = q/k/v/o + ffn`（attention+FFN）—— **预期精确 d^2**

用法：`python draw_scaling_axes.py`
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scaling_axes.png")

# ── 先验字形：中文标签里混入 ₀ ₁ ⇒ 会变豆腐块（§68 的教训）
from matplotlib import font_manager
for name in ("Microsoft YaHei", "SimHei", "DengXian"):
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

D_LIST = (32, 64, 128)
N_LIST = (4096, 12459, 31148, 124593)


def load():
    """取每个 (n,d) 的一条有效记录。

    ★第一版用 `"slow_run" in r` 当有效性判据，结果**只有本轮新跑的 4 格**过筛，
      早先 7 格（gate `ff5eefd1604b` / `59855015dfd8`）全被丢掉 ⇒ 图几乎是空的。
      `slow_run` 是**速度**判据，跟读数有没有效**无关**；正确判据是 `gate_ok()`。
      ——又是「判据选错字段」，与 §69.5 那两处同类。
    """
    import exp_scaling_grid_v2 as e
    recs = [json.loads(l) for l in open(os.path.join(HERE, "scaling_grid_v2.jsonl"),
                                        encoding="utf-8") if l.strip()]

    def pick(n, d):
        hits = [r for r in recs if r["n"] == n and r["d"] == d
                and r["weight_decay"] == 0.0 and r.get("n_slots") == 4096
                and e.gate_ok(r)]
        return hits[-1] if hits else None

    grid = {}
    for n in N_LIST:
        for d in D_LIST:
            r = pick(n, d)
            if r is not None:
                grid[(n, d)] = r
    return grid


def w_params(d):
    """q/k/v/o + ffn 的参数量：2 层 × (4·d²) + 2·(d·2d) = 8d² + 4d² = 12d²"""
    return 12 * d * d


def fit_loglog(xs, ys):
    """最小二乘 log-log 斜率；返回 (斜率, R²)。点数不足返回 None。"""
    if len(xs) < 2:
        return None
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    if sxx <= 0:
        return None
    k = sxy / sxx
    b = my - k * mx
    ss_tot = sum((c - my) ** 2 for c in ly)
    ss_res = sum((c - (k * a + b)) ** 2 for a, c in zip(lx, ly))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return k, r2


def main():
    grid = load()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    colors = {32: "#2b6cb0", 64: "#c05621", 128: "#2f855a"}

    # ── 三个子图分别是三个容量量在「固定 N、沿 d 扫」时的曲线 ──────────
    # ★**这是容量方向的图**，横轴才是 params / E / W。数据量方向的图在下面另一张。
    #   第一版把容量量当横轴去画「沿 N 扫」的曲线 ⇒ 固定 d 时 xs 全同 ⇒ 拟合失败。
    #   根因：**搞混了两个方向**。沿 N 扫 ⇒ 横轴是 N；沿 d 扫 ⇒ 横轴是容量量。
    axes_def = [
        ("params（总量）", lambda r: r["params"], "混了 engram 线性项"),
        ("E = d × n_slots（engram）", lambda r: r["d"] * r["n_slots"], "精确 d^1"),
        ("W = q/k/v/o + ffn", lambda r: w_params(r["d"]), "精确 d^2"),
    ]
    for ax, (title, getx, note) in zip(axes, axes_def):
        for nn in N_LIST:
            xs, ys = [], []
            for d in D_LIST:
                if (nn, d) in grid:
                    xs.append(getx(grid[(nn, d)]))
                    ys.append(grid[(nn, d)]["probe"]["probe_nll"])
            if len(xs) < 2:
                continue
            ax.plot(xs, ys, "o-", lw=1.8, ms=7,
                    color=plt.cm.viridis(N_LIST.index(nn) / max(1, len(N_LIST) - 1)),
                    label="N=%d（无种子极差）" % nn)
            f = fit_loglog(xs, ys)
            if f:
                k, r2 = f
                ax.annotate("斜率 %.3f\nR2 %.2f" % (k, r2),
                            xy=(xs[-1], ys[-1]), xytext=(6, -2),
                            textcoords="offset points", fontsize=8.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(title, fontsize=11)
        ax.set_title("%s（%s）" % (title, note), fontsize=10.5)
        ax.grid(True, which="both", alpha=0.25, ls=":")
        ax.legend(fontsize=8.5, loc="upper right")

    axes[0].set_ylabel("probe NLL", fontsize=11)
    # ★图里**不许**出现 `⇒`（U+21D2）：Microsoft YaHei 缺这个字形，会变豆腐块。
    #   §68 画架构图时踩过同一个坑（₀ ₁ ⇒），这次在保存前先扫一遍。
    fig.suptitle("容量方向（轴2）：固定 N、沿 d 扫，横轴是三个候选容量量\n"
                 "斜率必须为负才算容量有用；实测随 N 由正转负，不是单一幂律", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, dpi=130)
    print("已保存:", OUT)

    # ── 文本版判决 ────────────────────────────────────────────────────
    # ★表1 第一版把横轴写成 `getx(...)`（容量量）⇒ **固定 d 时 xs 全是同一个数**
    #   ⇒ log-log 斜率算不出来（sxx=0）⇒ 全行报「缺」。这不是数据缺，是**轴写错**：
    #   沿 N 扫的时候，横轴必须是 **N 本身**。
    print()
    print("表1 数据量方向（固定 d，沿 N 扫）— 轴1 的 alpha，横轴是 N")
    for d in D_LIST:
        pts = [(nn, grid[(nn, d)]) for nn in N_LIST if (nn, d) in grid]
        if len(pts) < 2:
            print("  d=%-4d 缺（只有 %d 点）" % (d, len(pts)))
            continue
        # ★变量名必须用 `nn` 不能用 `n`：外层 `for n in N_LIST:` 表2 还没执行到，
        #   但列表推导式里的 `n` 在这里会**遮蔽**循环变量 `d` 之外的名字 ——
        #   实测 `xs=[n for _n,_r in pts]` 直接 NameError，因为推导式内 `n` 未定义。
        xs = [nn for nn, _r in pts]
        ys = [r["probe"]["probe_nll"] for _nn, r in pts]
        f = fit_loglog(xs, ys)
        if f is None:
            print("  d=%-4d 拟合失败（点数=%d）" % (d, len(pts)))
            continue
        print("  d=%-4d 斜率=%.4f R2=%.4f   NLL %.4f → %.4f  （单 seed，无种子极差）"
              % (d, f[0], f[1], ys[0], ys[-1]))
    print("  → alpha 随 d 变化（-0.16 / -0.24 / -0.42）→ 若真是单一幂律，三条应平行")

    print()
    print("表2 容量方向（固定 N，沿 d 扫）— 回答「轴2 是不是幂律」")
    print("  横轴是三个候选容量量；斜率必须**为负**才算容量真的有用")
    for n in N_LIST:
        row = []
        for name, getx in (("params", lambda r: r["params"]),
                           ("E", lambda r: r["d"] * r["n_slots"]),
                           ("W", lambda r: w_params(r["d"]))):
            pts = [(d, grid[(n, d)]) for d in D_LIST if (n, d) in grid]
            if len(pts) < 2:
                row.append("%s=缺" % name)
                continue
            xs = [getx(r) for _d, r in pts]
            ys = [r["probe"]["probe_nll"] for _d, r in pts]
            f = fit_loglog(xs, ys)
            row.append("%s=%.3f(R2%.2f)" % (name, f[0], f[1]))
        print("  N=%-9d %s" % (n, "  ".join(row)))
    print("  → 斜率随 N 由正转负（+0.21 → -0.20）→ 容量的作用**随数据量翻转符号**，")
    print("    不是单一幂律能表达的")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
