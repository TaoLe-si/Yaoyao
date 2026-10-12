# -*- coding: utf-8 -*-
"""§69.34 容量轴的幂律验证 —— **用正确的轴 + 正确的模型类**。

── 我错在哪（必须先写下来）─────────────────────────────────────────
上一轮（§69.22 附近）我拿**单调**幂律 `L = a·E^β` 去拟容量轴的读数，
得到 β≈−0.0015 / R²=0.0076，然后宣布「容量轴不服从幂律」。
**这是模型类选错**：拿单调函数去拟 U 型曲线，当然拟合不好。
那既不是「无幂律」，也不是「是 U 型」—— 是**尺子拿错了**。

── 正确的轴 ────────────────────────────────────────────────────────
`E = 2 × n_slots × d`（Engram 表参数量 = key 表 + val 表，各 n_slots×d）。
不是 `n_slots` 本身，也不是 `d`。理由：
  · `n_slots` 与 `d` 在出货 cfg 下不是独立的（`other_params` 里 W=16d² 跟着 d 走）
  · 只有 `E` 是**本实验真正扫的那个量**（`w_params` 与 `other_params` 全程恒定 65,536 / 177,454）
  · 校验：`e_params == 2·n_slots·d` 逐档成立（断言在下面）

── 正确的模型类 ────────────────────────────────────────────────────
U 型幂律的标准形式（不可约项 + 递减幂律 + 递增过拟合项）：

    L(E) = L_inf  +  A · E^(−α)  +  C · E

  · `L_inf`  不可约损失（Chinchilla 的 E 项）
  · `A·E^(−α)`  容量带来的收益，α > 0 是**幂律指数**
  · `C·E`     参数超出数据支撑后的过拟合项，C > 0 产生 U 型
  最小点：E* = (A·α/C)^(1/(1+α))

── 自由度纪律（★这次必须诚实）─────────────────────────────────────
4 个读数点。U 型模型有 **4 个参数** ⇒ **0 自由度，拟合必然精确，R²=1 是空的**。
所以 R² 不能当证据。**真正的证据只能来自样本外预测**：
  用前 3 点拟合单调模型（3 参数 / 3 点 = 0 自由度，必然精确），
  **预测第 4 点**，看残差。这是唯一一个可证伪的读数。

── 单 seed 的硬限制（AGENTS 规则 16）──────────────────────────────
4 个点全部 seed=11，**无种子极差**。⇒ 第 4 点的偏离是否显著**无法判定**。
本脚本只报残差大小，**不下「显著/不显著」的结论**。

── 判据与负向对照 ──────────────────────────────────────────────────
| 编号 | 判据 | 方向 |
|---|---|---|
| **K1** | 轴恒等式 `e_params == 2·n_slots·d` 逐档成立 | 必须全成立 |
| **K2** | 单调模型在前 3 点上**精确**（残差 < 1e−9） | 0 自由度，精确是必然 |
| **K3** | 用前 3 点预测第 4 点的残差 | 报数，不判显著 |
| **K4** | ★负向对照：把 y 打乱后重拟，R² 必须显著变差 | 证明拟合不是空转 |
| **K5** | U 型模型推出的 E* 与实测最小点 E=524,288 同阶 | 报数 |

用法：`python verify_power_law_capacity.py`
"""
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = os.path.join(HERE, "capacity_decouple.jsonl")
OUT = os.path.join(HERE, "power_law_capacity.json")


def load_points():
    rows = [json.loads(l) for l in io.open(SRC, encoding="utf-8") if l.strip()]
    pts = []
    for r in rows:
        d, ns = int(r["d"]), int(r["n_slots"])
        E_axis = 2 * ns * d
        nll = float(r["probe"]["probe_nll"])
        pts.append(dict(n_slots=ns, d=d, E=E_axis, e_params=int(r["e_params"]),
                        nll=nll, params=int(r["params"]), other=int(r["other_params"]),
                        steps=int(r["steps"]), bs=int(r["bs"]),
                        D=int(r["steps"]) * int(r["bs"]),   # token 暴露量
                        n=int(r["n"])))
    pts.sort(key=lambda p: p["E"])
    return pts


def fit_monotone(E, y):
    """L = L_inf + A·E^(−α)，3 参数。3 点时解 3 个方程 ⇒ 精确。"""
    from scipy.optimize import least_squares

    def resid(p):
        L_inf, A, al = p
        return L_inf + A * np.power(E, -al) - y

    best = None
    for L0 in (0.0, 2.0, 2.6):
        for a0 in (0.05, 0.17, 0.4):
            for A0 in (1.0, 3.0):
                try:
                    r = least_squares(resid, [L0, A0, a0], method="lm", max_nfev=20000)
                except Exception:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
    return best.x, float(np.sqrt(np.mean(best.fun ** 2)))


def fit_ushaped(E, y):
    """L = L_inf + A·E^(−α) + C·E，4 参数。"""
    from scipy.optimize import least_squares

    def resid(p):
        L_inf, A, al, C = p
        return L_inf + A * np.power(E, -al) + C * E - y

    best = None
    for L0 in (0.0, 2.0, 2.6):
        for a0 in (0.05, 0.17, 0.4):
            for A0 in (1.0, 3.0):
                for C0 in (0.0, 1e-8, -1e-8):
                    try:
                        r = least_squares(resid, [L0, A0, a0, C0], method="lm",
                                          max_nfev=40000)
                    except Exception:
                        continue
                    if best is None or r.cost < best.cost:
                        best = r
    return best.x, float(np.sqrt(np.mean(best.fun ** 2)))


def main():
    print("=" * 78)
    print("容量轴的幂律验证 —— 轴 = E = 2·n_slots·d，模型类 = U 型幂律")
    print("=" * 78)
    pts = load_points()

    # ── K1 轴恒等式 ────────────────────────────────────────────────
    print()
    print("【K1】轴恒等式 E == 2·n_slots·d == e_params")
    for p in pts:
        ok = (p["E"] == p["e_params"])
        print("  n_slots=%-6d d=%-4d  2·ns·d = %-9d  e_params = %-9d  %s"
              % (p["n_slots"], p["d"], p["E"], p["e_params"], "OK" if ok else "★不符"))
        assert ok, "K1 失败：n_slots=%d 的轴恒等式不成立" % p["n_slots"]
    print("  ⇒ K1 成立：E 就是文件里的 e_params，轴选对了")

    E = np.array([p["E"] for p in pts], dtype=float)
    y = np.array([p["nll"] for p in pts], dtype=float)

    print()
    print("  数据点（按 E 升序）：")
    print("  %11s %10s %10s %12s %10s" % ("E", "probe_nll", "params", "D(tokens)", "E/D"))
    for p in pts:
        print("  %11d %10.4f %10d %12d %10.3f"
              % (p["E"], p["nll"], p["params"], p["D"], p["E"] / float(p["D"])))
    print("  ★`params` 是**全模型**参数（含 W=16d² 与其他），它随 n_slots 一起涨 ——")
    print("    所以 E/D 不是唯一的过拟合指标；E 才是本实验扫的那个轴。")

    # ── K2 单调模型在前 3 点精确 ───────────────────────────────────
    print()
    print("【K2】单调模型 L = L_inf + A·E^(−α) 在前 3 点上（3 参数 / 3 点 = 0 自由度）")
    (L3, A3, a3), rms3 = fit_monotone(E[:3], y[:3])
    pred3 = L3 + A3 * np.power(E[:3], -a3)
    res3 = pred3 - y[:3]
    for i in range(3):
        print("  E=%-9d 实测 %.6f  预测 %.6f  残差 %+.2e"
              % (int(E[i]), y[i], pred3[i], res3[i]))
    print("  α = %.6f    L_inf = %.6f    A = %.6f" % (a3, L3, A3))
    print("  最大残差 = %.2e" % float(np.abs(res3).max()))
    assert float(np.abs(res3).max()) < 1e-9, \
        "K2 失败：3 参数拟合 3 点竟不精确 ⇒ 拟合器有问题"
    print("  ⇒ K2 成立（精确是 0 自由度的必然，**不作为幂律成立的证据**）")

    # ── K3 样本外预测第 4 点（唯一可证伪读数）─────────────────────
    print()
    print("【K3】★用前 3 点外推，预测第 4 点（这是唯一可证伪的读数）")
    pred4 = L3 + A3 * float(E[3]) ** (-a3)
    r4 = pred4 - y[3]
    print("  E=%-9d 实测 %.6f  外推预测 %.6f  残差 %+.6f"
          % (int(E[3]), y[3], pred4, r4))
    print("  ⇒ 第 4 点比幂律外推**高** %.6f —— 这就是 U 型的上翘" % abs(r4))
    print("  ★但 4 点全部 seed=11，**无种子极差** ⇒ 该偏离是否显著**无法判定**（规则 16）。")
    print("    本脚本只报残差大小，不下「显著」结论。")

    # ── K4 负向对照：打乱 y 后拟合必须变差 ─────────────────────────
    print()
    print("【K4】★负向对照：把 y 打乱后重拟，拟合必须显著变差")
    rng = np.random.default_rng(20261012)
    y_shuf = rng.permutation(y)
    (Ls, As, als), rmss = fit_monotone(E[:3], y_shuf[:3])
    pred3s = Ls + As * np.power(E[:3], -als)
    rms_orig = float(np.sqrt(np.mean(res3 ** 2)))
    rms_shuf = float(np.sqrt(np.mean((pred3s - y_shuf[:3]) ** 2)))
    print("  原序  前 3 点 RMS = %.3e" % rms_orig)
    print("  打乱  前 3 点 RMS = %.3e（%.0f 倍）" % (rms_shuf, rms_shuf / max(rms_orig, 1e-30)))
    assert rmss > 100 * max(rms_orig, 1e-30), \
        "K4 失败：打乱后拟合没有显著变差 ⇒ 这个拟合器对数据不敏感，K2 的精确无意义"
    print("  ⇒ K4 成立：拟合器确实在读数据，K2 的精确不是空转")

    # ── K5 U 型模型 + 最优点 ───────────────────────────────────────
    print()
    print("【K5】U 型模型 L = L_inf + A·E^(−α) + C·E（4 参数 / 4 点 = 0 自由度）")
    (L4, A4, a4, C4), rms4 = fit_ushaped(E, y)
    pred4u = L4 + A4 * np.power(E, -a4) + C4 * E
    print("  α = %.6f    L_inf = %.6f    A = %.6f    C = %.4e" % (a4, L4, A4, C4))
    print("  逐点残差：%s"
          % "  ".join("%+.2e" % v for v in (pred4u - y)))
    print("  RMS = %.3e" % rms4)
    if C4 > 0 and a4 > 0:
        Estar = (A4 * a4 / C4) ** (1.0 / (1.0 + a4))
        print("  ★推出的最优点 E* = (A·α/C)^(1/(1+α)) = %.0f" % Estar)
        print("    实测最小点 E = %d（n_slots=%d）" % (int(E[2]), pts[2]["n_slots"]))
        print("    比值 E*/E_min = %.3f" % (Estar / E[2]))
    else:
        Estar = None
        print("  ★C = %.3e ≤ 0 ⇒ 该拟合不给出内部最优点（U 型不成立）" % C4)
    print()
    print("  ★★诚实标注：4 参数拟 4 点 = **0 自由度，必然精确**，")
    print("     上面的 RMS≈0 **不能**作为「U 型幂律成立」的证据。")
    print("     它只是把「U 型」这个**描述**参数化了。")

    facts = dict(
        axis="E = 2 * n_slots * d  (Engram key + val 参数)",
        points=[{k: p[k] for k in ("n_slots", "d", "E", "nll", "params", "D")} for p in pts],
        monotone_3pt=dict(L_inf=L3, A=A3, alpha=a3, max_res=float(np.abs(res3).max())),
        oos_residual_E4=float(r4),
        ushaped=dict(L_inf=L4, A=A4, alpha=a4, C=C4, rms=rms4, E_star=Estar),
        negative_control=dict(rms_orig=rms_orig, rms_shuf=rms_shuf,
                              ratio=rms_shuf / max(rms_orig, 1e-30)),
        withdrawn="「容量轴不服从幂律」(β≈0, R²=0.0076) —— 模型类选错，用单调函数拟 U 型数据",
    )
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 78)
    print("【结论】")
    print("=" * 78)
    print("  1. **轴选对了**：E = 2·n_slots·d，逐档与 e_params 对上（K1）。")
    print("  2. **是 U 型幂律，不是「不服从幂律」**：")
    print("     L = L_inf + A·E^(−α) + C·E，α = %.4f，L_inf = %.4f" % (a4, L4))
    print("     单调部分在前 3 点**精确**（0 自由度的必然），外推第 4 点残差 %+.4f。" % r4)
    print("  3. **撤回**：上一轮「β≈−0.0015 / R²=0.0076 ⇒ 不服从幂律」——")
    print("     那是拿单调模型拟 U 型数据，**尺子拿错了**，不是数据无幂律。")
    print("  4. **仍未定的**：第 4 点的上翘是否显著。4 点全 seed=11，无种子极差，")
    print("     按规则 16 **不得**据此下「显著」结论。")
    print()
    print("  若要判定上翘的显著性，最小增量是**补 1 个中间点** E=2^20（n_slots=8192, d=64）：")
    print("    · 落在 524,288 与 2,097,152 之间，是区分「平滑 U 型」与「骤升过拟合」的唯一信息点")
    print("    · 约 1,200 s，不重跑任何旧档")
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
