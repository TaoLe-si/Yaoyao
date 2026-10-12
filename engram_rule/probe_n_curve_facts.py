"""§69.52 探针：把两条线的读数合成一张 (steps, N) 网格，并检验三件事。

读数来源（只读这两个落盘文件，不重新训练）：
  · `n_sweep.jsonl`          —— §69.51 的 epochs≡40 对角线（4 格）
  · `n_steps_decoupled.jsonl` —— §69.52 的 steps≡462 横线（新格）

## 这个探针要回答的三个问题（都可失败）

**Q1 共线性是真的吗？** §69.51 用 `steps = ceil(40N/256)`，理论上 `steps/N` 应恒等于
`1/6.4 = 0.15625`（只有 ceil 的量化抖动）。若实测 `steps/N` 恒定 ⇒ 「数据量」与
「优化预算」在那一批读数里**不可分辨**，任何单轴曲线都不能归因给数据量。
可失败性：把 N 换成人工非比例序列时该量必须变化。

**Q2 train 曲线是幂律吗？** 幂律 `a·N^b` 是**单调**的，装不下内点极大值。
若实测序列里内点极大值个数 = 1 ⇒ 幂律**结构上不适用**（不是拟合不好）。
可失败性：人造单调序列上该计数必须为 0。

**Q3 解耦后曲线是什么形状？** 在 steps≡462 上 tok_acc 随 N 的走向。
· 单调降 ⇒ epochs≡40 那个峰是「步数 × 数据量」的合成产物（D2 成立）
· 不单调 ⇒ 右侧下降另有机制（首选嫌疑：n-gram 别名把 4096 个槽占满）

## 负向对照（规则 9/29：断言判据的输入真的变了）
N1 单调性判据在**反转序列**上必须翻转，且断言反转后的列表 ≠ 原列表
N2 2 因子拟合在**打乱 y**后必须显著变差（R² 掉下来）
N3 内点极大值计数在人造单调序列上必须为 0，且断言该序列确实单调

用法：`python probe_n_curve_facts.py`
"""
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # 训练在跑，本探针只占 CPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

SWEEP = os.path.join(HERE, "n_sweep.jsonl")
DEC = os.path.join(HERE, "n_steps_decoupled.jsonl")
OUT = os.path.join(HERE, "n_curve_facts.json")

D = 128
N_SLOTS = 4096
STEPS_FIXED = 462
BS = 256
EPOCHS_TARGET = 40.0
GRID_N = [738, 1476, 2952, 5904, 11808]


# ── 判据本体（都要能被下面的负向对照推翻） ──────────────────────────
def is_monotone_decreasing(v, tol=0.0):
    """严格单调降判据。tol 是**可配的相等容忍**，由测量的分辨率给出，不是拍脑袋。"""
    return all(v[i] >= v[i + 1] - tol for i in range(len(v) - 1))


def count_interior_maxima(v, tol=0.0):
    """内点极大值个数。幂律是单调的 ⇒ 这个数必须为 0 才可能拟合。"""
    c = 0
    for i in range(1, len(v) - 1):
        if v[i] >= v[i - 1] - tol and v[i] > v[i + 1] + tol:
            c += 1
    return c


def fmt(n):
    return "{:,}".format(int(n))


def wilson(k, n, z=1.96):
    """Wilson 得分区间。EM 是 k/n 的二项比例，直接比点估计会被分母骗。"""
    if n <= 0:
        return [0.0, 0.0]
    p = k / float(n)
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return [round(max(0.0, ctr - half), 4), round(min(1.0, ctr + half), 4)]


def ols(X, y):
    """最小二乘 + R² + 残差。返回 (beta, r2, resid)。X 已含常数列。"""
    import numpy as np
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
    pred = Xa @ beta
    resid = ya - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return [float(b) for b in beta], r2, [float(r) for r in resid]


def logit(p, eps=1e-9):
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


# ── 读数装载 ───────────────────────────────────────────────────────
def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_cells():
    """按 (n, steps) 去重成网格。锚点 (2952,462) 两条线共用 ⇒ 只算一次、不重复计数。"""
    cells = {}
    for r in load_jsonl(SWEEP):
        key = (r["n"], r["steps"])
        c = dict(r)
        c["source"] = "sweep"
        c["lines"] = ["epochs40"]
        cells[key] = c
    for r in load_jsonl(DEC):
        key = (r["n"], r["steps"])
        if key in cells:
            # 锚点：两条线在同一点相交。必须逐字段核对，确认是同一格而不是同名不同义。
            a = cells[key]
            diff = [k for k in ("d", "train_tok_acc", "train_em", "train_f1",
                                "held_tok_acc") if k in r and k in a and a[k] != r[k]]
            a.setdefault("anchor_diff", []).extend(diff)
            a["lines"] = sorted(set(a["lines"] + ["steps462"]))
            a["cross_checked"] = not diff
        else:
            c = dict(r)
            c["source"] = "decoupled"
            c["lines"] = ["steps462"]
            cells[key] = c
    return cells


# ── 线的归属：由步数规则推导，不用落盘标签 ──────────────────────
def on_epochs40(n, steps):
    """这一格在不在「epochs 恒 40」对角线上：steps == ceil(40n/256)。"""
    return steps == -(-int(EPOCHS_TARGET * n) // BS)


def on_steps462(n, steps):
    """这一格在不在「steps 恒 462」横线上。"""
    return steps == STEPS_FIXED


def line_cells(cells, pred, field):
    """按规则挑出某条线上的格，返回 (N 列表, 字段列表)，按 N 升序。

    ★为什么不用落盘里的 `lines` 标签：锚点 (2952,462) 因为「复用不重跑」
      从没进过 decoupled 的 jsonl，只会被标成 epochs40 ⇒ 标签法会把它漏出
      steps462 线（实测 Q3 只数到 1 格而不是 2 格）。线归属是**几何事实**，
      必须由步数规则推出来，否则标签与几何会各说各话。
    """
    cs = [c for c in cells.values() if pred(c["n"], c["steps"])]
    cs.sort(key=lambda c: c["n"])
    return [c["n"] for c in cs], [c[field] for c in cs]


# ── 槽位压力（不训练，只建表） ────────────────────────────────────
def alive_sweep(ns):
    """alive_slots 随 N 的走向。

    ★为什么不训练也能量：`build_engram_from_examples` 只做一次计数直方图
      （dialogue_small.py:351-357），与权重无关 ⇒ 这是**零训练成本**的读数。
    ★为什么重要：它检验 D3 的前提。表只有 4096 个槽，distinct n-gram 数一旦逼近它
      就发生别名碰撞 ⇒ 「记不下」不是因为行装不下，而是因为 n-gram 别名。
    """
    out = {}
    try:
        import model as M
        rows = E2.load_rows()
        tr, _hd = E2.split_rows(rows)
        cfg = dict(E2._CK["cfg"])
        cfg["d"], cfg["n_heads"], cfg["n_slots"] = D, D // E2.HEAD_DIM, N_SLOTS
        cfg["pos_abs_max"], cfg["use_qpool"] = 512, False
        for n in ns:
            sub = tr[:n]
            x, st, l = E2.encode_cached(sub, E2._TOK, E2.SEQ)
            torch.manual_seed(11)
            m = M.EngramRuleLM(M.RCfg(**cfg))
            b = E2.DS.build_engram_from_examples(m, x, l)
            # ★为什么还要记 tokens_per_slot：实测 alive_ratio 在 N=738 就是 4091/4096
            #   （99.9%），一路到 N=11808 都是全满 ⇒ 它是**常量指标，没有区分度**，
            #   按规则 9 不能当证据。真正随 N 变的是「每个槽平均要承载多少 n-gram 出现」：
            #   tokens_per_slot = answer token 总数 / n_slots（build_engram 只数真 token）。
            ans_tok = int((l - st).sum())
            out[str(n)] = dict(alive_slots=int(b["alive_slots"]),
                               alive_ratio=round(float(b["alive_ratio"]), 6),
                               answer_tokens=ans_tok,
                               tokens_per_slot=round(ans_tok / float(N_SLOTS), 3))
            print("   alive N=%-6d %d/%d (%.2f%%)  answer_tokens=%d  %.2f token/槽"
                  % (n, b["alive_slots"], N_SLOTS, 100.0 * b["alive_ratio"],
                     ans_tok, ans_tok / float(N_SLOTS)))
            del m
    except Exception as exc:                       # 失败要**留痕**，不许静默吞
        out["_error"] = "%s: %s" % (type(exc).__name__, exc)
        print("   !! alive 扫描失败（留痕）：%s" % out["_error"])
    return out


# ── 主流程 ─────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("§69.52 探针：N 曲线事实")
    print("=" * 78)
    print("  torch %s" % torch.__version__)
    # ★训练正在占 GPU，本探针**不许**碰显存：显式把 E2.DEV 钉成 cpu。
    #   （实测 `CUDA_VISIBLE_DEVICES=""` 在本机 torch 2.14 上仍报 cuda=True，
    #     所以不能靠那个环境变量 —— 之前那句「强制 CPU」的打印是假的，一并撤掉。）
    E2.DEV = "cpu"
    print("  本探针 DEV 钉为 cpu（训练在跑，不抢显存）；torch.cuda.is_available()=%s"
          % torch.cuda.is_available())

    # ★必须先 load_all 再碰 _CK：它是模块全局 None，`load_all()` 内部用 `global`
    #   赋值（exp_scaling_grid_v2.py:781）。先读就是 NoneType 下标 ⇒ 这就是
    #   「函数体依赖的全局只在别处赋值」那类坑（AGENTS §1A 的 C 步）。
    E2.load_all()
    assert E2._CK is not None and E2._TOK is not None, "load_all 后 _CK/_TOK 仍为空"

    cells = build_cells()
    if not cells:
        raise SystemExit("neither n_sweep.jsonl nor n_steps_decoupled.jsonl has cells")
    print("  网格 %d 格（按 (n,steps) 去重）" % len(cells))

    # 参数量：三个数分开列（state_dict ⊇ 训练参数，buffer 也落盘）
    import model as M
    cfg = dict(E2._CK["cfg"])
    cfg["d"], cfg["n_heads"], cfg["n_slots"] = D, D // E2.HEAD_DIM, N_SLOTS
    cfg["pos_abs_max"], cfg["use_qpool"] = 512, False
    m = M.EngramRuleLM(M.RCfg(**cfg))
    n_train_params = sum(p.numel() for p in m.parameters())
    n_buf = sum(b.numel() for b in m.buffers())
    sd = m.state_dict()
    n_sd = sum(v.numel() for v in sd.values())
    assert n_train_params + n_buf == n_sd, \
        "参数口径不自洽：%d+%d != %d" % (n_train_params, n_buf, n_sd)
    assert set(named for named, _ in m.named_parameters()) | \
        set(named for named, _ in m.named_buffers()) == set(sd.keys()), \
        "named_parameters|buffers 与 state_dict 键集不一致"
    print("  参数：可训练 %d / buffer %d / state_dict %d（恒等式自检通过）"
          % (n_train_params, n_buf, n_sd))

    # ── Q1 共线性 ────────────────────────────────────────────────
    diag = sorted([c for c in cells.values() if on_epochs40(c["n"], c["steps"])],
                  key=lambda c: c["n"])
    ratios = [c["steps"] / float(c["n"]) for c in diag]
    # ★共线性的推导：steps_i = ceil(40·n_i/256) ⇒ steps_i/n_i = 40/256 + e_i/n_i，
    #   其中 e_i ∈ [0,1)。于是每个比值 ∈ [40/256, 40/256 + 1/n_i)，
    #   整组极差 **严格小于 1/min_n**。这就是判据的门槛，由被测量本身推出。
    const_ratio = EPOCHS_TARGET / BS          # = 40/256 = 0.15625（不是它的倒数）
    spread_bound = 1.0 / float(min(c["n"] for c in diag))
    q1 = dict(
        diagonal_n=[c["n"] for c in diag],
        diagonal_steps=[c["steps"] for c in diag],
        steps_over_n=[round(r, 6) for r in ratios],
        ratio_spread=round(max(ratios) - min(ratios), 6),
        predicted_constant=round(const_ratio, 6),
        predicted_spread_bound=round(spread_bound, 6),
        bound_derivation="steps/n = 40/256 + e/n, e in [0,1) => spread < 1/min_n",
    )
    q1["is_collinear"] = q1["ratio_spread"] < q1["predicted_spread_bound"]
    print()
    print("  Q1 steps/N = %s" % q1["steps_over_n"])
    print("     预测常量 40/256 = %.6f；极差上界 1/min_n = %.6f；实测极差 %.6f"
          % (q1["predicted_constant"], q1["predicted_spread_bound"],
             q1["ratio_spread"]))
    print("     ⇒ 共线性 %s（成立意味着「数据量」与「优化预算」在该批读数里不可分辨）"
          % ("成立" if q1["is_collinear"] else "不成立"))

    # ── Q2 幂律结构 ──────────────────────────────────────────────
    ns_e, tr_e = line_cells(cells, on_epochs40, "train_tok_acc")
    hd_e = line_cells(cells, on_epochs40, "held_tok_acc")[1]
    n_inter = count_interior_maxima(tr_e)
    q2 = dict(line="epochs40", n=ns_e, train_tok_acc=tr_e,
              interior_maxima=n_inter,
              peak_n=ns_e[tr_e.index(max(tr_e))],
              monotone_decreasing=is_monotone_decreasing(tr_e),
              held_tok_acc=hd_e,
              held_peak_n=ns_e[hd_e.index(max(hd_e))],
              held_monotone_increasing=all(hd_e[i] <= hd_e[i + 1]
                                           for i in range(len(hd_e) - 1)))
    print()
    print("  Q2 epochs40 上 train_tok_acc = %s" % ["%.4f" % v for v in tr_e])
    print("     内点极大值个数 = %d ⇒ 幂律（单调）%s；峰在 N=%d"
          % (n_inter, "结构上不适用" if n_inter else "未被排除", q2["peak_n"]))

    # ── Q3 解耦线 ────────────────────────────────────────────────
    ns_s, tr_s = line_cells(cells, on_steps462, "train_tok_acc")
    q3 = dict(line="steps462", n=ns_s, train_tok_acc=tr_s)
    if len(tr_s) >= 2:
        q3["monotone_decreasing"] = is_monotone_decreasing(tr_s)
        q3["interior_maxima"] = count_interior_maxima(tr_s)
        q3["peak_n"] = ns_s[tr_s.index(max(tr_s))]
        # 幂律斜率（仅在 ≥3 点上给，且同时报 logit 版）
        if len(tr_s) >= 3:
            lx = [math.log(v) for v in ns_s]
            ly = [math.log(max(v, 1e-9)) for v in tr_s]
            b, r2, res = ols([[1.0, v] for v in lx], ly)
            q3["powerlaw"] = dict(intercept=b[0], slope=b[1], r2=r2, residuals=res)
        print()
        print("  Q3 steps462 上 train_tok_acc = %s（n=%d 点）"
              % (["%.4f" % v for v in tr_s], len(tr_s)))
        if q3["monotone_decreasing"]:
            print("     单调降 ⇒ D1 成立：epochs40 的峰是**步数×数据量的合成产物**")
        else:
            print("     不单调降 ⇒ D1 被推翻：右侧下降另有机制（先查槽位饱和）")
    else:
        print()
        print("  Q3 steps462 线还没跑出 ≥2 格（现有 %d 格）" % len(tr_s))

    # ── 2 因子拟合：把「数据量」与「优化预算」分开 ──────────────────
    keys = sorted(cells.keys(), key=lambda k: (k[1], k[0]))
    # ★门槛 6 格不是我拍的：3 个参数要留 dof>=3 才能让排列零分布有分辨力。
    #   4 格时 dof=1，实测「y 轮转一位」的 R² 反而更高（0.6958 -> 0.9969），
    #   证明单一扰动在 dof=1 下毫无判别力。宁可不做，也不报一个没有功效的 R²。
    if len(keys) >= 6:
        feats = [[1.0, math.log(cells[k]["steps"]), math.log(k[0])] for k in keys]
        b_t, r2_t, res_t = ols(feats, [cells[k]["train_tok_acc"] for k in keys])
        b_h, r2_h, res_h = ols(feats, [cells[k]["held_tok_acc"] for k in keys])
        g_t, r2_g, res_g = ols(feats, [logit(cells[k]["train_tok_acc"]) for k in keys])
        fit = dict(
            cells=[[k[0], k[1]] for k in keys],
            n_cells=len(keys), dof=len(keys) - 3,
            train=dict(intercept=b_t[0], b_log_steps=b_t[1], b_log_n=b_t[2],
                       r2=r2_t, residuals=res_t),
            held=dict(intercept=b_h[0], b_log_steps=b_h[1], b_log_n=b_h[2],
                      r2=r2_h, residuals=res_h),
            train_logit=dict(intercept=g_t[0], b_log_steps=g_t[1], b_log_n=g_t[2],
                             r2=r2_g, residuals=res_g),
            exchange_rate=(round(-b_t[2] / b_t[1], 4) if b_t[1] else None),
        )
        fit["exchange_rate_note"] = (
            "保持 train tok_acc 不变时，log N 每 +1 需要 log steps 变化 "
            "%.4f（负号表示二者互相替代）" % fit["exchange_rate"]
            if fit["exchange_rate"] is not None else "b_log_steps=0，无法换算")
        print()
        print("  2 因子拟合 train_tok_acc ~ 1 + log steps + log N（%d 格，dof=%d）"
              % (len(keys), len(keys) - 3))
        print("     b_log_steps=%.4f  b_log_n=%.4f  R²=%.4f"
              % (b_t[1], b_t[2], r2_t))
        print("     logit 版：b_log_steps=%.4f  b_log_n=%.4f  R²=%.4f"
              % (g_t[1], g_t[2], r2_g))

        import itertools
        ys = [cells[k]["train_tok_acc"] for k in keys]
        perms = list(itertools.permutations(ys))
        null_r2 = sorted(ols(feats, list(p))[1] for p in perms)
        rank = sum(1 for v in null_r2 if v >= r2_t)
        p95 = null_r2[max(0, int(0.95 * (len(null_r2) - 1)))]
        perm_ok = bool(r2_t > p95)
        fit["perm_test"] = dict(n_perms=len(null_r2), real_r2=r2_t,
                                null_p95=p95, rank_desc=rank,
                                best_null=null_r2[-1], worst_null=null_r2[0])
        fit["perm_ok"] = perm_ok
        # ★**这个模型族被拒绝，不是「拟合稍差」**，所以只留档、不再往下用。
        #   实测（6 格）：真实 R²=0.3473，而 y 全 720 排列的零分布最高能到 0.9853、
        #   p95=0.9052 —— 真实拟合排第 381 高。原因是**设定错误**：对数平面上的
        #   平面装不下复合轴上的内点极大值（Q2 已证幂律结构上不适用），于是任何
        #   「碰巧单调」的排列都能被这个平面拟合得更好。
        fit["verdict"] = ("ACCEPTED" if perm_ok else
                          "REJECTED_BY_PERMUTATION: 设定错误（平面装不下内点极大值），"
                          "不用它的系数，也不用它的 R²")
        print()
        print("  [尝试后拒绝] train_tok_acc ~ 1 + log steps + log N（%d 格，dof=%d）"
              % (len(keys), len(keys) - 3))
        print("     b_log_steps=%.4f  b_log_n=%.4f  R²=%.4f（logit 版 R²=%.4f）"
              % (b_t[1], b_t[2], r2_t, r2_g))
        print("     排列零分布（y 全 %d 排列）：[%.4f, %.4f]，p95=%.4f，"
              "真实排第 %d 高"
              % (len(null_r2), null_r2[0], null_r2[-1], p95, rank))
        print("     ⇒ %s" % fit["verdict"])
    else:
        fit = dict(error="格子不足 6（现有 %d），dof 不足 ⇒ 不做 2 因子拟合，"
                         "也不报 R²" % len(keys))
        print("  2 因子拟合：现有 %d 格 < 6，跳过（宁可不做，也不报无功效的 R²）"
              % len(keys))

    # ── 模型无关的配对对比：同 N 换 steps / 同 steps 换 N ──────────
    # ★为什么不用拟合来回答「数据量 vs 最优」：上面的检验已经证明这个格子数下
    #   参数拟合**没有推断功效**。配对对比不需要任何模型 —— 两格之间只差一个因子，
    #   差值本身就是那个因子的效应。这是唯一不依赖曲线族的读法。
    pairs = []
    for ka in keys:
        for kb in keys:
            if ka >= kb:
                continue
            ca, cb = cells[ka], cells[kb]
            same_n = ka[0] == kb[0]
            same_steps = ka[1] == kb[1]
            if not (same_n or same_steps):
                continue
            iso = "只变 steps" if same_n else "只变 N"
            pairs.append(dict(
                a=[ka[0], ka[1]], b=[kb[0], kb[1]], isolates=iso,
                d_train_tok_acc=round(cb["train_tok_acc"] - ca["train_tok_acc"], 5),
                d_train_em=round(cb["train_em"] / float(cb["train_em_n"])
                                - ca["train_em"] / float(ca["train_em_n"]), 5),
                d_held_tok_acc=round(cb["held_tok_acc"] - ca["held_tok_acc"], 5)))
    fit_paired = dict(contrasts=pairs,
                      note="差值 = (b - a)；「只变 steps」的对比里 N 恒定，"
                           "「只变 N」的对比里 steps 恒定，因此差值可直接读作该因子的效应")
    print()
    print("  配对对比（模型无关，差值即效应）：")
    for p in pairs:
        print("     N %s->%s steps %d->%d [%s]  d_train_tok_acc=%+.4f  "
              "d_train_em=%+.4f  d_held_tok_acc=%+.4f"
              % (fmt(p["a"][0]), fmt(p["b"][0]), p["a"][1], p["b"][1],
                 p["isolates"], p["d_train_tok_acc"], p["d_train_em"],
                 p["d_held_tok_acc"]))

    # ── 负向对照 N1 / N3 ────────────────────────────────────────
    # ★N1 的设计我第一版写错了：拿「反转序列」当对照。可**非单调**序列反转之后
    #   仍然非单调 ⇒ 判定永远不会翻转 ⇒ 对照必然"失效"。
    #   （教训：对照不触发时先怀疑对照选错，别急着怀疑判据 —— 这条已进记忆。）
    #   正确对照：构造一个**确实单调降**的序列，判据必须返回 True；而真实序列返回
    #   False。两者成对出现，才证明「判据能失败」。
    mono_probe = sorted(tr_e, reverse=True)
    assert mono_probe != tr_e, "N1 对照构造失败：与真实序列相同 ⇒ 判据输入没变"
    mono_on_probe = is_monotone_decreasing(mono_probe)
    mono_real = is_monotone_decreasing(tr_e)
    mono_series = sorted(tr_e)                      # 人造严格单调升
    n_real = count_interior_maxima(tr_e)
    n_mono = count_interior_maxima(mono_series)
    neg = dict(
        N1=dict(real=mono_real, on_sorted_desc=mono_on_probe,
                can_fail=(mono_on_probe != mono_real),
                probe_series=mono_probe, input_changed=True),
        N3=dict(real=n_real, on_monotone_series=n_mono,
                count_can_fail=(n_real != 0), monotone_series=mono_series),
    )
    print()
    print("  负向对照 N1：单调性判据 真实序列=%s 人工单调降=%s ⇒ %s"
          % (mono_real, mono_on_probe,
             "可失败" if neg["N1"]["can_fail"] else "**恒真，判据失效**"))
    print("  负向对照 N3：内点极大值计数 真实=%d 人造单调=%d ⇒ %s"
          % (n_real, n_mono, "可失败" if n_mono == 0 else "**恒非零，判据失效**"))
    assert neg["N1"]["can_fail"], "N1 负向对照失败：单调性判据在任何输入上同一判定"
    assert n_mono == 0, "N3 负向对照失败：内点极大值计数在单调序列上不为 0"

    # ── EM 的 Wilson 区间（判「最优点」用，不比点估计） ───────────
    em_tab = []
    for k in keys:
        c = cells[k]
        if "train_em" in c:
            em_tab.append(dict(n=k[0], steps=k[1], em=c["train_em"],
                               em_n=c["train_em_n"],
                               p=round(c["train_em"] / float(c["train_em_n"]), 4),
                               ci95=wilson(c["train_em"], c["train_em_n"]),
                               on_epochs40=on_epochs40(k[0], k[1]),
                               on_steps462=on_steps462(k[0], k[1])))
    print()
    print("  train EM（Wilson 95%%）——★推荐点只在 **epochs 恒 40 扫描线**内选，")
    print("     因为它是 §69.51 的扫描口径；steps≡462 那条线是诊断用的，混排是跨口径。")
    for e in sorted(em_tab, key=lambda z: -z["p"]):
        print("     N=%-6d steps=%-4d %s EM=%3d/%d = %.3f  CI95=[%.3f, %.3f]"
              % (e["n"], e["steps"], "[扫描线]" if e["on_epochs40"] else "[诊断线]",
                 e["em"], e["em_n"], e["p"], e["ci95"][0], e["ci95"][1]))

    # ── 派生量 ──────────────────────────────────────────────────
    for k in keys:
        c = cells[k]
        c["params_per_row"] = round(n_train_params / float(k[0]), 2)
        c["slot_load"] = round(k[0] / float(N_SLOTS), 4)
        c["epochs"] = round(BS * c["steps"] / float(k[0]), 3)

    print()
    print("  槽位饱和度扫描（零训练成本）：")
    alive = alive_sweep(GRID_N)

    facts = dict(
        meta=dict(d=D, n_slots=N_SLOTS, bs=BS, epochs_target=EPOCHS_TARGET,
                  steps_fixed=STEPS_FIXED, seed=11, held_probe=2000,
                  train_params=n_train_params, buffers=n_buf, state_dict=n_sd,
                  probe_device=E2.DEV),
        cells=[[k[0], k[1], cells[k]] for k in keys],
        q1_collinearity=q1, q2_powerlaw=q2, q3_decoupled=q3,
        fit_two_factor=fit, paired=fit_paired, negative_controls=neg,
        em_table=em_tab, alive_sweep=alive,
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())