"""§66: 严格复核已经落盘的二维网格，不训练、不运行模型前向或 eval。

历史 §63 的五参数拟合及 LOO 散布不能提供置信区间。正向的
L(N,d)=E+A/N^alpha+B/d^beta (A,B,alpha,beta>0) 至少要求固定一轴时
另一轴的损失非增。这里先检查该必要条件；通过也不等于证明幂律。
历史公式及被撤回的推论保留在 DESIGN.md §63/§66，不再作为判决程序。
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "scaling_grid_v1.jsonl")


def audit_grid(records, *, gate, ns, ds, recipe):
    """完整格点和同口径门禁后，返回两轴的逐格对照；不估计种子误差。"""
    ns, ds = tuple(sorted(ns)), tuple(sorted(ds))
    if (len(ns) < 2 or len(ds) < 2 or len(set(ns)) != len(ns)
            or len(set(ds)) != len(ds)):
        raise ValueError("N 和 d 各需至少两个互异档位")
    if len(records) != len(ns) * len(ds):
        raise ValueError("格点数不等于预登记的完整网格")

    grid = {}
    for r in records:
        if r.get("gate") != gate:
            raise ValueError("旧 gate 或未核实来源的记录")
        if any(r.get(k) != value for k, value in recipe.items()):
            raise ValueError("格点的训练/探针口径不一致")
        key = (r.get("n"), r.get("d"))
        if key[0] not in ns or key[1] not in ds or key in grid:
            raise ValueError("重复或额外的 (N,d) 格点")
        try:
            nll = float(r["probe_nll"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("缺少有效 probe_nll") from exc
        if not math.isfinite(nll) or nll < 0:
            raise ValueError("probe_nll 必须为非负有限数")
        grid[key] = nll
    if set(grid) != {(n, d) for n in ns for d in ds}:
        raise ValueError("缺失预登记格点")

    capacity_adjacent = {n: [grid[n, d1] - grid[n, d0]
                             for d0, d1 in zip(ds, ds[1:])] for n in ns}
    capacity_end = {n: grid[n, ds[-1]] - grid[n, ds[0]] for n in ns}
    data_adjacent = {d: [grid[n1, d] - grid[n0, d]
                         for n0, n1 in zip(ns, ns[1:])] for d in ds}
    return dict(
        grid_cells=len(grid),
        capacity_adjacent_deltas=capacity_adjacent,
        capacity_end_deltas=capacity_end,
        data_adjacent_deltas=data_adjacent,
        capacity_nonincreasing=all(v <= 0 for values in capacity_adjacent.values()
                                   for v in values),
        data_nonincreasing=all(v <= 0 for values in data_adjacent.values()
                               for v in values),
        nll_grid=grid,
    )


def main():
    import exp_scaling_grid as exp

    with open(SRC, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    ns, ds = exp.N_LIST, exp.D_LIST
    # 5,200 是 §63 固定 82 行探针的已落盘 token 数；它不由当前代码反算。
    recipe = dict(steps=exp.STEPS, seed=exp.SEED, bs=exp.BS,
                  n_probe=exp.N_PROBE, probe_tokens=5200,
                  lr=exp.LR, head_dim=exp.HEAD_DIM, micro_batch=64)
    result = audit_grid(records, gate=exp.GATE, ns=ns, ds=ds, recipe=recipe)
    grid = result["nll_grid"]

    print("§66 只读复核: %d/%d 格完整, gate=%s, seed=%d (无种子极差)"
          % (result["grid_cells"], len(ns) * len(ds), exp.GATE, exp.SEED))
    print("历史固定探针: %d 行 / %d tokens；本次没有新训练或 eval"
          % (exp.N_PROBE, recipe["probe_tokens"]))
    print("N       " + "  ".join("d=%-3d" % d for d in ds) + "  Δ(d=%d−%d)" % (ds[-1], ds[0]))
    for n in sorted(ns):
        print("%-6d  %s  %+.6f"
              % (n, "  ".join("%.6f" % grid[n, d] for d in sorted(ds)),
                 result["capacity_end_deltas"][n]))
    print("固定 N 的逐档容量 Δ: %s" % result["capacity_adjacent_deltas"])
    print("固定 d 的逐档数据 Δ: %s" % result["data_adjacent_deltas"])
    print("N 倍率 %.2f, log10 跨度 %.3f" % (max(ns) / min(ns), math.log10(max(ns) / min(ns))))
    print("必要条件: 容量非增=%s; 数据非增=%s"
          % (result["capacity_nonincreasing"], result["data_nonincreasing"]))
    if not (result["capacity_nonincreasing"] and result["data_nonincreasing"]):
        print("结论: 这 15 格不支持正向的双轴 scaling law；单 seed 无种子极差，不作总体显著性推断。")
    else:
        print("结论: 两个单调必要条件成立；仍不足以证明跨尺度幂律或 compute-optimal law。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
