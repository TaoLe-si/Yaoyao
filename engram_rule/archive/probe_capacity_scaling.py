"""★实验 A 的判决脚本: 读原始 JSONL, 按 DESIGN §48(含修订 A)的预登记规则出判决。

**只读**, 不训练。判据全部来自 §48, 这里不新增任何判据 —— 跑的人不许顺手改尺子。

  C0 预检 : lr 的极差 vs 同档种子噪声          (§48)
  P3      : 某档 train 答案 NLL 比 d=64 **更差** > 2SE ⇒ 该档优化坏了, 作废
  ★主判据 : S(d) = β(d) − β(64),  β = hit ~ log(1+LOO) 的 OLS 斜率(§48 修订 A)
  次要     : R3 = ΔR1 − ΔR2 (功效不足已实测, **不得单独用于判决**)

**S 的负向对照 (规则 7)**: 把 LOO 在行内随机置换后重算 β 应 ≈0。
若仍显著非 0 ⇒ 这个量测的不是"频次依赖", S 无效, 拒绝下结论。

**规则 12 硬闸**: 任一比较若种子数 < 2 或 |差| ≤ 2×SE, 一律输出"无差异, 不得下结论"。
"""
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "capacity_scaling_v1.jsonl")
SE_MULT = 2.0
MIN_SEEDS = 2
NC_REPS = 20
NC_SEED = 1234


def load():
    if not os.path.exists(SRC):
        raise SystemExit("找不到 %s —— 先跑 exp_capacity_scaling.py" % SRC)
    out = []
    with open(SRC, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def agg(vals):
    n = len(vals)
    m = statistics.fmean(vals)
    if n < MIN_SEEDS:
        return m, float("nan"), n
    if n == 2:
        return m, abs(vals[0] - vals[1]) / 2.0, n
    return m, statistics.stdev(vals) / math.sqrt(n), n


def fmt(v, w=9, p=4):
    return ("%*.*f" % (w, p, v)) if v == v else "%*s" % (w, "n/a")


def verdict_line(label, diff, se):
    if se != se:
        return "%s: 无种子极差(n<%d) -> **不得下结论**" % (label, MIN_SEEDS)
    thr = SE_MULT * se
    if abs(diff) > thr:
        return "%s: %+.4f  > 2SE=%.4f  => **可区分**" % (label, diff, thr)
    return "%s: %+.4f <= 2SE=%.4f  => 无差异, **不得下结论**" % (label, diff, thr)


# --------------------------------------------------------------------- 斜率

def slope(hit, loo):
    xs = [math.log1p(float(v)) for v in loo]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(hit) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, hit))
    return sxy / sxx


def slope_negative_control(hit, loo, reps=NC_REPS):
    """把 LOO 在行内随机置换后重算斜率。**必须 ≈0**, 否则 S 测的不是频次依赖。"""
    rnd = random.Random(NC_SEED)
    vals = []
    for _ in range(reps):
        p = list(loo)
        rnd.shuffle(p)
        v = slope(hit, p)
        if v == v:
            vals.append(v)
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    return (statistics.fmean(vals),
            statistics.stdev(vals) / math.sqrt(len(vals)),
            max(abs(v) for v in vals))


# --------------------------------------------------------------------- C0

def report_c0(recs):
    rows = [r for r in recs if r["arm"] == "precheck"]
    if not rows:
        print("\n=== C0 lr 敏感性预检: 尚未跑, 跳过 ===")
        return None
    print("\n=== C0 lr 敏感性预检 (d=%d) ===" % rows[0]["d"])
    by_lr = defaultdict(list)
    for r in rows:
        by_lr[r["lr"]].append(r)
    print("  %-8s %-6s %10s %10s %12s %12s %8s"
          % ("lr", "n", "R1(LOO0)", "R2(LOO4)", "train NLL", "1stAcc", "Δtrain"))
    for lr in sorted(by_lr):
        g = by_lr[lr]
        r1, _, n = agg([x["r1_loo0_acc"] for x in g])
        r2, _, _ = agg([x["r2_loo4_acc"] for x in g])
        nl, _, _ = agg([x["train"]["answer_nll"] for x in g])
        fa, _, _ = agg([x["train"]["first_answer_token_accuracy"] for x in g])
        dt, _, _ = agg([x["delta_train"] for x in g])
        print("  %-8s %-6d %s %s %s %s %s"
              % (lr, n, fmt(r1), fmt(r2), fmt(nl), fmt(fa), fmt(dt)))

    r1_means = {lr: agg([x["r1_loo0_acc"] for x in g])[0] for lr, g in by_lr.items()}
    ses = [agg([x["r1_loo0_acc"] for x in g])[1] for g in by_lr.values()]
    ses = [s for s in ses if s == s]
    if len(r1_means) < 2 or not ses:
        print("  ★C0 未完成(至少需要 2 个 lr 且每档 >=%d seed) -> 无法判定" % MIN_SEEDS)
        return None
    spread = max(r1_means.values()) - min(r1_means.values())
    noise = max(ses)
    print("  R1 跨 lr 极差 = %.4f ; 同 lr 内种子噪声(取最大 SE) = %.4f" % (spread, noise))
    if spread > SE_MULT * noise:
        print("  ★判定: **lr 敏感**(极差 %.4f > 2SE %.4f) => 正式实验须每档取 lr 最优, 成本 ×3"
              % (spread, SE_MULT * noise))
        return "sensitive"
    print("  ★判定: lr 不敏感(极差 %.4f <= 2SE %.4f) => 正式实验固定 lr=0.003"
          % (spread, SE_MULT * noise))
    return "insensitive"


# --------------------------------------------------------------------- 主判决

def report_main(recs):
    rows = [r for r in recs if r["arm"] == "main"]
    if not rows:
        print("\n=== 主实验: 尚未跑 ===")
        return
    need = [r for r in rows if "hit_all" not in r]
    if need:
        print("\n=== 主实验: %d 条缺 hit_all(逐行命中率), 判据 S 无法计算 ===" % len(need))
        return

    by_d = defaultdict(list)
    for r in rows:
        by_d[r["d"]].append(r)
    ds = sorted(by_d)
    if 64 not in ds:
        print("\n=== 主实验: 缺 d=64 基线档, 无法算 S ===")
        return

    print("\n=== 主实验: 每档逐 run 原始读数 ===")
    print("  %-5s %-5s %-6s %-6s %9s %9s %10s %10s %9s %9s %9s"
          % ("d", "seed", "mb", "param", "β(斜率)", "R1(LOO0)", "R2(LOO4)",
             "trainNLL", "1stAcc", "Δtrain", "Δeval"))
    beta = {}
    for d in ds:
        for r in sorted(by_d[d], key=lambda x: x["seed"]):
            b = slope(r["hit_all"], r["loo_all"])
            beta[(d, r["seed"])] = b
            print("  %-5d %-5d %-6d %-6d %s %s %s %s %s %s %s"
                  % (d, r["seed"], r["micro_batch"], r["params_total"], fmt(b),
                     fmt(r["r1_loo0_acc"]), fmt(r["r2_loo4_acc"]),
                     fmt(r["train"]["answer_nll"]),
                     fmt(r["train"]["first_answer_token_accuracy"]),
                     fmt(r["delta_train"]), fmt(r["delta_eval"])))

    # ── S 的负向对照 (规则 7) ───────────────────────────────────────────
    print("\n=== S 的负向对照 (LOO 行内随机置换 %d 次, 斜率应 ≈0) ===" % NC_REPS)
    nc_ok = True
    for d in ds:
        r0 = sorted(by_d[d], key=lambda x: x["seed"])[0]
        mu, se, mx = slope_negative_control(r0["hit_all"], r0["loo_all"])
        bad = (se == se) and abs(mu) > SE_MULT * se
        nc_ok = nc_ok and not bad
        print("  d=%-4d seed=%-3d 置换后 mean=%+.5f SE=%.5f max|b|=%.5f  %s"
              % (d, r0["seed"], mu, se, mx, "★显著非 0 ⇒ S 无效" if bad else "通过"))
    if not nc_ok:
        print("\n  ★负向对照未过 ⇒ S 测的不是频次依赖, **拒绝下结论**。")
        return

    # ── 跨 seed 汇总 ────────────────────────────────────────────────────
    seeds = sorted({r["seed"] for r in rows})
    per = {d: {s: next(r for r in by_d[d] if r["seed"] == s) for s in seeds
               if any(x["seed"] == s for x in by_d[d])} for d in ds}

    print("\n=== 跨 seed 汇总 (d=64 为基线档) ===")
    print("  %-5s %-4s %9s %9s %9s %9s %9s %12s"
          % ("d", "n", "β", "SE", "★S=β−β64", "SE", "R3(次要)", "trainNLL"))
    summary = {}
    for d in ds:
        bs = [beta[(d, s)] for s in per[d]]
        b, bse, n = agg(bs)
        svals, r3vals, nlls = [], [], []
        for s in per[d]:
            if s in per[64]:
                svals.append(beta[(d, s)] - beta[(64, s)])
                a_, b_ = per[d][s], per[64][s]
                r3vals.append((a_["r1_loo0_acc"] - b_["r1_loo0_acc"])
                              - (a_["r2_loo4_acc"] - b_["r2_loo4_acc"]))
        sv, sse, _ = agg(svals)
        r3v, r3se, _ = agg(r3vals)
        nllv, nllse, _ = agg([per[d][s]["train"]["answer_nll"] for s in per[d]])
        summary[d] = dict(beta=b, beta_se=bse, S=sv, S_se=sse, R3=r3v, R3_se=r3se,
                          nll=nllv, nll_se=nllse,
                          params=per[d][sorted(per[d])[0]]["params_total"],
                          engram=per[d][sorted(per[d])[0]]["engram_pct"])
        print("  %-5d %-4d %s %s %s %s %s %s   %d / %.1f%%"
              % (d, n, fmt(b), fmt(bse), fmt(sv), fmt(sse), fmt(r3v), fmt(nllv),
                 summary[d]["params"], summary[d]["engram"]))

    print("\n=== 预登记判决 (DESIGN §48 + 修订 A) ===")
    base = summary[64]
    broken = []
    for d in ds:
        if d == 64:
            continue
        s = summary[d]
        if s["nll_se"] == s["nll_se"] and s["nll"] - base["nll"] > SE_MULT * s["nll_se"]:
            broken.append(d)
            print("  P3 命中: d=%d train NLL=%.4f 比 d=64 的 %.4f **更差**(>2SE=%.4f)"
                  " ⇒ 该档**优化坏了**, 作废, 不得据此判 P2"
                  % (d, s["nll"], base["nll"], SE_MULT * s["nll_se"]))
    if not broken:
        print("  P3: 无档位触发(NLL 均未显著变差) => 没有「优化坏了」的档")

    print()
    for d in ds:
        if d == 64:
            continue
        s = summary[d]
        # P1 判 S 显著为**负**(频次依赖被消除); R3 判显著为**正**(低频组涨更多)。
        print("  d=%-4d %s" % (d, verdict_line("★S 主判据(命中=显著为负)",
                                               -s["S"], s["S_se"])))
        print("  %s%s" % (" " * 7, verdict_line("R3 次要(命中=显著为正)",
                                               s["R3"], s["R3_se"])))

    usable = [d for d in ds if d != 64 and d not in broken]
    # ★符号: β = slope(hit ~ log(1+LOO))。β>0 = 命中率**依赖**出现频次 = 学不会低频。
    #   容量治好低频 ⇒ β 下降 ⇒ **S 显著为负**才是 P1。
    #   ⚠ 本脚本第一版把 P1 写成 `S > 2SE`(照抄了 R3 的正方向), 因 S 恒为负而误报 P2。
    #     这是**判据写错**, 不是数据问题(规则 5)。
    hit = [d for d in usable if summary[d]["S_se"] == summary[d]["S_se"]
           and -summary[d]["S"] > SE_MULT * summary[d]["S_se"]]
    contra = [d for d in usable if summary[d]["R3_se"] == summary[d]["R3_se"]
              and abs(summary[d]["R3"]) > SE_MULT * summary[d]["R3_se"]
              and ((-summary[d]["S"] > SE_MULT * summary[d]["S_se"])
                   != (summary[d]["R3"] > SE_MULT * summary[d]["R3_se"]))]
    print()
    if hit:
        print("  ★命中 **P1**: d=%s 的 S **显著为负** ⇒ β 被压到 0 附近(频次依赖消失)"
              " ⇒ **容量是约束**, 且低频端受益多于高频端" % hit)
        print("    ⇒ §46 的「提高每个开头频次」只是**缓解**; 正解是放大容量。")
    else:
        und = [d for d in usable if summary[d]["S_se"] != summary[d]["S_se"]]
        if und:
            print("  ★**无法判决**: d=%s 缺种子极差 ⇒ 依规则 12 不得下结论(需补 seed)" % und)
        else:
            print("  ★命中 **P2**: 无档位 S 显著为负 ⇒ **容量不是约束**, 问题在训练动态")
            print("    ⇒ 回到 §46 的待验假设 `first_token_weight`。")
    if contra:
        print("  ⚠ 修订 A 第 3 条: d=%s 上 **S 与 R3 方向相反** ⇒ 按预登记**两个都要报**"
              % contra)

    # ── 迁移率: Δeval/Δtrain 随容量是否变化(§42 的 21% 是基线) ──────────
    #  SE 用 delta method: r = e/t, dr/de = 1/t, dr/dt = −e/t², Δtrain/Δeval 按 seed 配对。
    print("\n=== 迁移率 r = Δeval/Δtrain (随容量是否变化?) ===")
    print("  %-5s %4s %10s %10s %10s %10s %12s"
          % ("d", "n", "Δtrain", "SE", "Δeval", "SE", "r=e/t ± SE"))
    ratios = {}
    for d in ds:
        t = [per[d][s]["delta_train"] for s in per[d]]
        e = [per[d][s]["delta_eval"] for s in per[d]]
        tm, ts, n = agg(t)
        em, es, _ = agg(e)
        r = em / tm
        if ts == ts and es == es and n >= MIN_SEEDS:
            var = ((es / tm) ** 2) + ((em * ts / (tm * tm)) ** 2)
            rse = math.sqrt(var)
        else:
            rse = float("nan")
        ratios[d] = (r, rse)
        print("  %-5d %4d %s %s %s %s %s"
              % (d, n, fmt(tm), fmt(ts), fmt(em), fmt(es), fmt(r) + " ± " + fmt(rse)))
    print()
    for d in ds:
        if d == 64:
            continue
        r, rs = ratios[d]
        r0, r0s = ratios[64]
        diff = r - r0
        se = math.sqrt(rs ** 2 + r0s ** 2) if (rs == rs and r0s == r0s) else float("nan")
        print("  d=%-4d %s" % (d, verdict_line("迁移率变化 r(d)−r(64)", diff, se)))


def main():
    recs = load()
    print("读入 %d 条 run 记录 (%s)" % (len(recs), os.path.basename(SRC)))
    report_c0(recs)
    report_main(recs)
    return 0


if __name__ == "__main__":
    sys.exit(main())