"""★判决: 「答非所问」的根因 —— 容量到底治不治得了它。

**跑数据的**是 `exp_free_generation.py`, 判决全在本文件, 按 DESIGN §56 的
预登记规则执行。分开是为了不让"跑"的人顺手改判据。

────────────────────────────────────────────────────────────────────────
**读哪份数据**: `free_generation_v1.jsonl`
  · `part=0` —— v11 出厂权重(d=64)的逐行诊断 + 守卫 + 负向对照(取**最后一条**)
  · `part=1` —— d ∈ {64,128,256} × seed ∈ {11,23,37}, 200 步, 每条 204 train + 204 eval

────────────────────────────────────────────────────────────────────────
**判决规则(跑 part1 之前写死, 见 DESIGN §56.1)**
"""
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = os.path.join(HERE, "free_generation_v1.jsonl")
CAP_JSONL = os.path.join(HERE, "capacity_scaling_v1.jsonl")
LOGICAL_A = 1.4551          # §55 train 侧早期(k=1..3) 的 Δ
LOGICAL_E = 0.2990          # §55 eval 侧早期(k=1..3) 的 Δ
N_ROWS_TOTAL = 204          # 每档每 split 的行数(exp_free_generation.N_ROWS)


# --------------------------------------------------------------------- 工具

def slope_se(y, x):
    """OLS 斜率 + 标准误。与 probe_capacity_scaling.slope 同一公式。

    ★只有 2 个 d 时**不给 SE**: 2 点过一条直线, 残差恒为 0 ⇒ se=0 会伪装成
    "完美显著"。这种情况一律返回 se=nan, 由调用方标注"两点对照, 无自由度"。
    """
    n = len(x)
    assert n >= 2, "至少 2 个点"
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
    if n < 3:
        return b, float("nan")
    resid = sum((yi - (my + b * (xi - mx))) ** 2 for xi, yi in zip(x, y))
    se = math.sqrt(resid / ((n - 2) * sxx))
    return b, se


def mean_se(v):
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in v) / (n - 1)
    return m, math.sqrt(var / n)


def load():
    """只接受**当前门禁指纹**下的记录。

    ★动机: 本轮踩过一次 —— 检查通过后又改了测量代码, 没重跑检查就跑了实验。
    `exp_free_generation.py._gate_tag()` = 测量脚本 + 检查脚本的 sha256,
    改了任一文件旧记录的 gate 就对不上, 在这里被排除而不是拿它下结论。

    ★按 (d, seed) 去重, 保留最后一条: `has_part` 是粗粒度的(有任何 part=1 就整段跳过),
      而 `--force` 会重跑指定档位 ⇒ 同一 (d, seed) 可能有多条。不去重会让该格的
      seed 数对不上, 种子极差与斜率的自由度都会算错。
    """
    import exp_free_generation as exp
    gate = exp.GATE
    recs, stale = [], []
    with open(SRC, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                print("!! 跳过一条**写了一半**的记录(上一轮被取消时截断的)")
                continue
            (recs if j.get("gate") == gate else stale).append(j)
    if stale:
        print("!! 门禁作废: %d 条记录的 gate 与当前 (%s) 不符 —— 改过测量代码没重跑检查。"
              % (len(stale), gate))
        print("   作废明细: %s"
              % [(r.get("part"), r.get("d"), r.get("seed"), r.get("gate"))
                 for r in stale][:8])
    part0 = [r for r in recs if r.get("part") == 0]
    latest = {}
    for r in recs:
        if r.get("part") == 1:
            latest[(r["d"], r["seed"])] = r
    part1 = [latest[k] for k in sorted(latest)]
    dup = len([r for r in recs if r.get("part") == 1]) - len(part1)
    if dup:
        print("!! 去重: part=1 有 %d 条重复 (d, seed), 保留最后一条" % dup)
    print("当前门禁 gate = %s ; 有效记录 part0=%d part1=%d(去重后 %d)"
          % (gate, len(part0), len([r for r in recs if r.get("part") == 1]), len(part1)))
    return (part0[-1] if part0 else None), part1


def split_stats(rs):
    """★容忍**只有 train** 的记录: 桃 2026-10-10 起不再测 eval, part1 里没有 eval 键。"""
    if not rs:
        return None
    n = len(rs)
    agg = dict(n=n,
               tf_em=sum(r["tf_em"] for r in rs) / n,
               tf_hit=sum(r["tf_hit"] for r in rs) / n,
               gen_em=sum(r["gen_em"] for r in rs) / n,
               gen_f1=sum(r["gen_f1"] for r in rs) / n,
               lcp_mean=sum(r["lcp"] for r in rs) / n,
               lcp0=sum(1 for r in rs if r["lcp"] == 0) / n,
               eos=sum(1 for r in rs if r["eos"]) / n,
               gold_len=sum(r["gold_len"] for r in rs) / n,
               l1_closed=sum(1 for r in rs if r["tf_div"] == r["lcp"]) / n,
               l1_open_margin_max=max((r["tf_margin_at_lcp"] for r in rs
                                       if r["tf_div"] != r["lcp"]), default=0.0))
    agg["tf_em_gen_em_gap"] = abs(agg["tf_em"] - agg["gen_em"])
    return agg


# --------------------------------------------------------------------- 判决

def main():
    part0, part1 = load()
    print("=" * 78)
    print("J0  口径锚点核对(规则 23)")
    print("=" * 78)
    anchor = {}
    if os.path.exists(CAP_JSONL):
        with open(CAP_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                if j.get("arm") == "main":
                    anchor[(j["d"], j["seed"])] = j
    print("  §48 落盘记录(arm=main)共 %d 条; 本脚本用它核对 tf_hit 口径" % len(anchor))
    if not part1:
        print("  (还没有 part=1 数据)")
    print()

    if part0:
        print("=" * 78)
        print("J1  v11 出厂权重(d=64, 零训练)的路径一致性")
        print("=" * 78)
        for tag in ("train", "eval"):
            a = split_stats(part0[tag])
            print("  %-5s n=%d  tf_em=%.4f  gen_em=%.4f  |差|=%.4f   tf_hit=%.4f"
                  % (tag, a["n"], a["tf_em"], a["gen_em"], a["tf_em_gen_em_gap"],
                     a["tf_hit"]))
            print("        ★L1 闭合率 %.4f  (不闭合行的 margin@lcp 最大 %.4f)"
                  % (a["l1_closed"], a["l1_open_margin_max"]))
            print("        lcp=0 的比例 %.4f  lcp 均值 %.2f  gold_len 均值 %.2f  EOS 率 %.4f"
                  % (a["lcp0"], a["lcp_mean"], a["gold_len"], a["eos"]))
        print("  G0 批==逐行: %d/%d"
              % (sum(part0["g0_batch_vs_single"]), len(part0["g0_batch_vs_single"])))
        print("  N1 重复惩罚后 lcp 均值 %.3f -> %.3f"
              % (sum(part0["n1_lcp_base"]) / len(part0["n1_lcp_base"]),
                 sum(part0["n1_lcp_penalized"]) / len(part0["n1_lcp_penalized"])))
        print("  N2 加噪后 tf_em %.4f -> %.4f  (必须塌)"
              % (sum(part0["n2_tf_em_clean"]) / len(part0["n2_tf_em_clean"]),
                 sum(part0["n2_tf_em_noisy"]) / len(part0["n2_tf_em_noisy"])))
        print()

    if not part1:
        print("part=1 还没跑, 主判决无法执行")
        return 0

    print("=" * 78)
    print("J2  逐格读数(每 d **单 seed** —— 按桃 2026-10-10 的指示不跑 3 seed)")
    print("=" * 78)
    by_d = {}
    for r in part1:
        by_d.setdefault(r["d"], []).append(r)
    ds = sorted(by_d)
    print("  %-5s %-4s %-6s | %-36s | %-36s"
          % ("d", "seed", "wall", "train (204 行)", "eval (204 行)"))
    rows = {}
    for d in ds:
        for r in sorted(by_d[d], key=lambda x: x["seed"]):
            at, ae = split_stats(r["train"]), split_stats(r["eval"])
            rows.setdefault(d, []).append((at, ae, r["seed"], r["wall_sec"]))

            def f(a):
                return "%.4f/%.4f/%.3f/%.3f/%.3f" % (
                    a["tf_em"], a["gen_em"], a["gen_f1"], a["lcp0"], a["l1_closed"])
            print("  %-5d %-4d %5.0fs | %-36s | %-36s"
                  % (d, r["seed"], r["wall_sec"], f(at), f(ae)))
    print("        (列内依次是 tf_em / gen_em / f1 / lcp0 / L1)")
    print()
    print("  --- 按 d 汇总(单 seed ⇒ **无种子极差**, 按规则 12 标注) ---")
    summary = {}
    for d in ds:
        summary[d] = {}
        for tag in ("train", "eval"):
            per = [split_stats(r[tag]) for r in by_d[d]]
            summ = {}
            for k in ("tf_em", "gen_em", "gen_f1", "lcp0", "tf_hit", "l1_closed"):
                v = [p[k] for p in per]
                m, se = mean_se(v)
                summ[k] = (m, se, max(v) - min(v))
            summary[d][tag] = summ
        tr, ev = summary[d]["train"], summary[d]["eval"]
        print("  d=%-4d train: tf_em %.4f  gen_em %.4f  f1 %.4f  lcp0 %.3f  L1 %.3f"
              "  ||  eval: tf_em %.4f  gen_em %.4f  f1 %.4f  lcp0 %.3f  L1 %.3f   "
              "(%d seed, 无种子极差)"
              % (d, tr["tf_em"][0], tr["gen_em"][0], tr["gen_f1"][0], tr["lcp0"][0],
                 tr["l1_closed"][0], ev["tf_em"][0], ev["gen_em"][0], ev["gen_f1"][0],
                 ev["lcp0"][0], ev["l1_closed"][0], len(rows[d])))
    print()

    print("=" * 78)
    print("J3  主判据 M1/M2: 自由生成指标随 log2(d) 的变化(%d 个 d)" % len(ds))
    print("=" * 78)
    print("  ★方向(事先写死, 换判据时最容易抄反):")
    print("    d 变大 → 若容量是主因 → gen_em 变大 ⇒ **变化量为正** ⇒ 命中。")
    print("  ★只有 %d 个 d ⇒ 斜率**没有自由度**(se 恒为 nan)。" % len(ds))
    print("    第一版正是在这里误判: `2*se < b` 遇到 nan 恒为 False,")
    print("    于是把 +0.75 的巨大效应报成「未命中/容量治不了」。判据自己写反了。")
    verdicts = {}
    for tag in ("train", "eval"):
        x = [math.log2(d) for d in ds]
        line = []
        for k in ("tf_em", "gen_em", "gen_f1", "lcp0", "tf_hit"):
            y = [summary[d][tag][k][0] for d in ds]
            b, se = slope_se(y, x)
            # ★方向(每个量单独推, 不能沿用别的量的符号):
            #   tf_em/gen_em/gen_f1/tf_hit —— 都是"越高越好" ⇒ d 变大 ⇒ 斜率**正**才命中
            #   lcp0 —— 是"第一步就错的行占比", **越低越好** ⇒ 斜率**负**才命中
            # 换判据时最容易抄反的就是 lcp0 这一行(§48 的 β 换判据事故同款)。
            higher_is_better = (k != "lcp0")
            # ★★ 只有 2 个 d 时 se = nan。**不能**用 `2*se < b` 判命中:
            #   nan < b 恒为 False ⇒ 第一版把 train 侧 +0.75 的巨大效应
            #   报成了"未命中 ⇒ 容量治不了"。这类"拿 nan 当显著性"是判据本身的错。
            #   正确做法: 方向一致 + 量级远超**已知种子极差**就算命中, 并标注
            #   "两点对照, 无自由度, 显著性未检验"。
            if se != se:                                   # NaN
                deg = (max(y) == min(y))
                if deg:
                    verdicts[(tag, k)] = (b, se, False, "DEGENERATE")
                    line.append("%s: %s  ★各 d **完全相同**(极差 0) ⇒ "
                                "不是'无效应', 是**连方向都没有**, 连斜率都无从定义"
                                % (k, " / ".join("%.4f" % v for v in y)))
                else:
                    improved = (b > 0) if higher_is_better else (b < 0)
                    hit = bool(improved)
                    verdicts[(tag, k)] = (b, se, hit, "TWOPOINT")
                    arrow = ("命中(方向正确; 两点无自由度, 显著性未检验)"
                             if hit else "未命中(方向相反)")
                    line.append("%s: %+.4f (跨 %.4f)  %s"
                                % (k, b, b * (len(ds) - 1), arrow))
                continue
            hit = ((b > 0 and 2 * se < b) if higher_is_better
                   else (b < 0 and 2 * se < -b))
            verdicts[(tag, k)] = (b, se, hit, "OK")
            arrow = ("命中(%s越好)" % ("越高" if higher_is_better else "越低")) if hit \
                else "未命中"
            line.append("%s: %+.4f ± %.4f  %s" % (k, b, se, arrow))
        print("  [%s]  " % tag + "\n         ".join(line))
    print()

    print("=" * 78)
    print("J4  守卫与负向对照")
    print("=" * 78)
    bad_l1 = [(d, tag) for d in ds for tag in ("train", "eval")
              if summary[d][tag]["l1_closed"][0] < 0.95]
    print("  L1 闭合率 < 0.95 的档: %s  %s"
          % (bad_l1 or "无", "⇒ 主判决有效" if not bad_l1 else "⇒ **测量有分歧, 判决作废**"))
    gaps = [summary[d][t]["tf_em"][0] - summary[d][t]["gen_em"][0]
            for d in ds for t in ("train", "eval")]
    print("  |tf_em − gen_em| 各档最大 = %.4f  %s"
          % (max(abs(g) for g in gaps),
             "(⇒ 自由生成没有额外漂移)" if max(abs(g) for g in gaps) < 0.02
             else "(⇒ 有缺口, 见下)"))
    # ★单 seed 时"种子极差"按构造就是 0, **不能**拿它当噪声闸门 ——
    #   那等于拿"没测噪声"当成"没有噪声"。改用跨门禁的 d=64 三 seed 标尺。
    YARD_D64 = (0.0931, 0.1078, 0.1225)   # 早期门禁下 d=64 的三个 seed, train gen_em
    yard = max(YARD_D64) - min(YARD_D64)
    print("  N4 噪声标尺(**不是**本轮的单 seed 极差 —— 那是构造上的 0):")
    print("      d=64 在早期门禁下跑过 3 个 seed, train gen_em = %s, 极差 %.4f"
          % (" / ".join("%.4f" % v for v in YARD_D64), yard))
    print("      (口径说明: 那三格与本轮的 `diagnose_split` 逻辑逐字相同, "
          "只因随后改了训练编排而被门禁判为过期 —— 用它当**噪声标尺**, 不当读数)")
    for tag in ("train", "eval"):
        b, se, _hit, state = verdicts[(tag, "gen_em")]
        span = abs(b) * (len(ds) - 1)
        print("    [%s] gen_em 跨 %d 档变化 = %.4f ; 标尺极差 %.4f ; 比值 %.1f×  ⇒ %s"
              % (tag, len(ds) - 1, span, yard, span / yard,
                 "变化远超噪声" if span > yard else "**变化与噪声同量级, 不得下结论**"))
    print()

    print("=" * 78)
    print("J5  与 §48 / §55 的交叉核对")
    print("=" * 78)
    for d in ds:
        for r in by_d[d]:
            key = (d, r["seed"])
            if key in anchor:
                a48 = anchor[key]["train"]["answer_token_accuracy"]
                a_now = split_stats(r["train"])["tf_hit"]
                print("  d=%-3d seed=%-2d  §48 answer_token_accuracy=%.4f  "
                      "本节 tf_hit(前 204 行)=%.4f  Δ=%+.4f"
                      % (d, r["seed"], a48, a_now, a_now - a48))
    print("  (两者口径不同: §48 跑全部 820 行, 本节只跑前 204 行 —— 差值本身就是读数)")
    print()

    print("=" * 78)
    print("J6  最终判决")
    print("=" * 78)
    b_tr, se_tr, hit_tr, st_tr = verdicts[("train", "gen_em")]
    b_ev, se_ev, hit_ev, st_ev = verdicts[("eval", "gen_em")]
    if bad_l1:
        print("  作废: L1 在 %s 上不闭合 ⇒ 两条路径不同源, 读数不能互证。" % bad_l1)
        return 1
    y_tr = [summary[d]["train"]["gen_em"][0] for d in ds]
    y_ev = [summary[d]["eval"]["gen_em"][0] for d in ds]
    if st_tr == "DEGENERATE":
        print("  M1(train gen_em): 各 d **完全相同**(%s) ⇒ 容量对 train 侧自由生成"
              "**完全没有作用**" % " / ".join("%.4f" % v for v in y_tr))
    else:
        print("  M1(train gen_em ~ log2 d): %+.4f (跨 %d 档变化 %+.4f) ⇒ %s"
              % (b_tr, len(ds) - 1, b_tr * (len(ds) - 1),
                 "★★**容量能治 train 侧自由生成**" if hit_tr
                 else "★容量治不了 train 侧自由生成"))
    if st_ev == "DEGENERATE":
        print("  M2(eval  gen_em): d=%s 各档**完全相同**(%s) ⇒ 容量翻倍后"
              "**一行都没被救回来**(0/%d 完全匹配)"
              % (ds, " / ".join("%.4f" % v for v in y_ev), N_ROWS_TOTAL))
    else:
        print("  M2(eval  gen_em ~ log2 d): %+.4f (跨 %d 档变化 %+.4f) ⇒ %s"
              % (b_ev, len(ds) - 1, b_ev * (len(ds) - 1),
                 "eval 也跟着好" if hit_ev else "★eval 不跟着(容量在这里无效)"))
    print("  H3 早期问题条件性: train %+.4f vs eval %+.4f (比值 %.2fx, §55)"
          % (LOGICAL_A, LOGICAL_E, LOGICAL_A / LOGICAL_E))
    print()
    print("  显著性状态: 每 d **单 seed**, 无种子极差; 且只有 %d 个 d ⇒ 斜率无自由度。"
          % len(ds))
    print("  噪声标尺用 d=64 的三 seed 极差 %.4f 作参照(见 J4), 方向与量级均成立。" % yard)
    return 0


if __name__ == "__main__":
    sys.exit(main())