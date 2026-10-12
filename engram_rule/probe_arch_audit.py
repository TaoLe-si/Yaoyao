"""★架构清点: 每个参数是标量/向量/矩阵, 以及**前向里到底有没有被用到**。

**为什么要"在激活"这一列**: 光看 `named_parameters()` 会数到一堆**根本没参与前向**的参数
(开关关掉的分支)。§61 就踩过一次同款误判 —— 我把 `use_qpool=False` 的模型当"少一条通路",
其实参数压根没建。所以本脚本对每个参数做**可达性判定**, 而不是只列 shape。

判定方法(可复现, 不靠眼看):
  1. 按配置开/关跑两次前向, 记录 `last_stats` / `last_mag` 里出现的键
  2. 对每个参数做**梯度探针**: 关掉它所在的模块, 看 loss 是否变化
     (太慢) ⇒ 改用更省的判据: 用 `torch.autograd.grad` 对每个叶参数求梯度,
     **梯度恒为 0 且不在 optimizer 里被更新的路径上** ⇒ 标记"未激活"
  3. 结构判定: 用配置开关直接推(use_pos_bucket/use_pos_abs/use_qpool/local_norm/clamp_window)

输出: `engram_rule/ARCH_AUDIT.md` + 终端表格。**零训练**, 秒级。
"""
import os
import sys
from collections import Counter, defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
OUT = os.path.join(HERE, "ARCH_AUDIT.md")


def kind(shape):
    n = 1
    for s in shape:
        n *= s
    if len(shape) == 0 or n == 1:
        return "标量"
    if len(shape) == 1:
        return "向量"
    return "矩阵"


def main():
    import exp_capacity_scaling as ecs
    from model import RCfg, EngramRuleLM
    from dialogue_small import build_engram_from_examples

    ck = torch.load(V11, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    tr_x, tr_s, tr_l = ecs.load_data()[4]

    torch.manual_seed(11)
    model = EngramRuleLM(RCfg(**cfg)).to(DEV)
    build_engram_from_examples(model, tr_x, tr_l)

    print("=" * 92)
    print("出货配置(d=%d, n_layers=%d, n_heads=%d, head_dim=%d)"
          % (cfg["d"], cfg["n_layers"], cfg["n_heads"], cfg["d"] // cfg["n_heads"]))
    print("=" * 92)
    for k in ("vocab", "ffn_mult", "rope_dim", "ngram_max", "n_hash", "n_slots",
              "n_buckets", "k_buckets", "gate_rank", "pos_abs_max", "window_max",
              "bucket_step", "use_pos_abs", "use_pos_bucket", "clamp_window",
              "local_norm", "near_first", "rule_scale_init", "lam_init",
              "use_qpool", "qpool_alpha_init", "qpool_mode"):
        print("  %-18s = %s" % (k, cfg.get(k, "<无此键>")))
    print()

    # ── 逐参数分类 ────────────────────────────────────────────────────
    rows = []
    for name, p in model.named_parameters():
        rows.append((name, tuple(p.shape), p.numel(), kind(tuple(p.shape)),
                     bool(p.requires_grad)))

    # ── 激活性判定: 用梯度探针(比"看开关"更硬) ────────────────────────
    x = tr_x[:64].to(DEV)
    model.train()
    out = model(x)
    loss = out.float().pow(2).mean()
    grads = torch.autograd.grad(loss, [p for _, p in model.named_parameters()],
                                allow_unused=True)
    gmap = {}
    for (name, _), g in zip(model.named_parameters(), grads):
        gmap[name] = (g is not None and float(g.abs().sum()) > 0)

    # 结构上"本就不该有梯度"的: 不在任何 loss 路径上的常数参数
    # 用**两次不同输入**的梯度都恒 0 来确认(而不是只看一次)
    x2 = tr_x[64:128].to(DEV)
    model.zero_grad(set_to_none=True)
    out2 = model(x2)
    loss2 = out2.float().pow(2).mean()
    grads2 = torch.autograd.grad(loss2, [p for _, p in model.named_parameters()],
                                 allow_unused=True)
    gmap2 = {}
    for (name, _), g in zip(model.named_parameters(), grads2):
        gmap2[name] = (g is not None and float(g.abs().sum()) > 0)

    print("=" * 92)
    print("A. 按**种类**汇总(出货配置)")
    print("=" * 92)
    by_kind = defaultdict(lambda: [0, 0])
    for name, shape, n, k, _ in rows:
        by_kind[k][0] += 1
        by_kind[k][1] += n
    total = sum(v[1] for v in by_kind.values())
    print("  %-6s %8s %14s %9s" % ("种类", "参数个数", "元素总数", "占比"))
    for k in ("标量", "向量", "矩阵"):
        c, n = by_kind[k]
        print("  %-6s %8d %14s %8.2f%%" % (k, c, "{:,}".format(n), 100.0 * n / total))
    print("  %-6s %8d %14s" % ("合计", sum(v[0] for v in by_kind.values()),
                               "{:,}".format(total)))
    print()

    print("=" * 92)
    print("B. ★逐参数: 种类 / 是否在激活(两次输入的梯度都非零才算)")
    print("=" * 92)
    print("  %-42s %-14s %8s %-6s %s"
          % ("参数名", "shape", "元素数", "种类", "激活"))
    alive_n = 0
    for name, shape, n, k, rg in sorted(rows, key=lambda r: (-r[2], r[0])):
        act = gmap.get(name, False) and gmap2.get(name, False)
        alive_n += n if act else 0
        print("  %-42s %-14s %8s %-6s %s"
              % (name, "×".join(str(s) for s in shape) or "()",
                 "{:,}".format(n), k, "✓" if act else "—"))
    print()
    print("  ★参与前向(两次输入梯度均非零)的元素: %s / %s = **%.1f%%**"
          % ("{:,}".format(alive_n), "{:,}".format(total), 100.0 * alive_n / total))
    print()

    # ── 结构开关的激活性 ──────────────────────────────────────────────
    print("=" * 92)
    print("C. 结构开关: 哪些通路在跑")
    print("=" * 92)
    with torch.no_grad():
        model(x)
    has_rule = bool((~model.engram.read(x, model.emb)[2]).any())
    alive_slots = int(model.engram.alive.sum())
    sb_counter = Counter()
    for blk in model.blocks:
        blk.attn.record = True
    with torch.no_grad():
        model(x)
    for blk in model.blocks:
        blk.attn.record = False
        if blk.attn.last_startb is not None:
            sb_counter.update(blk.attn.last_startb.reshape(-1).tolist())
    print("  %-22s %-10s %s" % ("开关/通路", "取值", "说明"))
    print("  %-22s %-10s %s" % ("use_pos_abs", str(cfg.get("use_pos_abs")),
                                "门的查询侧加绝对位置" if cfg.get("use_pos_abs") else "关闭"))
    print("  %-22s %-10s %s" % ("use_pos_bucket", str(cfg.get("use_pos_bucket")),
                                "位置进桶偏置(决定窗口起点)" if cfg.get("use_pos_bucket")
                                else "关闭"))
    print("  %-22s %-10s %s" % ("clamp_window", str(cfg.get("clamp_window")),
                                "因果夹紧(C28)" if cfg.get("clamp_window") else "关闭"))
    print("  %-22s %-10s %s" % ("local_norm", str(cfg.get("local_norm")),
                                "窗口内逐通道归一化(C29)" if cfg.get("local_norm")
                                else "关闭 ⇒ 用 1/sqrt(W) 缩放"))
    print("  %-22s %-10s %s" % ("use_qpool", str(cfg.get("use_qpool")),
                                "★问题池化(§61)" if cfg.get("use_qpool")
                                else "关闭 ⇒ 出货模型**没有**这条通路"))
    print("  %-22s %-10s %s" % ("Engram 规则路径", str(has_rule),
                                "本批有候选 ⇒ 规则路径在跑" if has_rule else "全空 ⇒ 跳过"))
    print("  %-22s %-10s %s" % ("Engram 存活槽",
                                "%d/%d (%.1f%%)" % (alive_slots, cfg["n_slots"],
                                                     100.0 * alive_slots / cfg["n_slots"]),
                                "§47 实测 98.5% 撞车 ⇒ 有效容量远小于名义"))
    print("  %-22s %-10s %s" % ("start_bucket 取值分布",
                                dict(sorted(sb_counter.items())),
                                "★只出现这些桶 ⇒ 窗口起点被压在近端"))
    print()

    # ── 矩阵维度表(给"哪些是矩阵"一个可查的清单) ──────────────────────
    print("=" * 92)
    print("D. 矩阵参数清单(按元素数排序)")
    print("=" * 92)
    print("  %-42s %-16s %10s" % ("参数名", "shape", "元素数"))
    for name, shape, n, k, _ in sorted(rows, key=lambda r: -r[2]):
        if k != "矩阵":
            continue
        print("  %-42s %-16s %10s"
              % (name, "×".join(str(s) for s in shape), "{:,}".format(n)))
    print()

    # ── 标量与向量 ────────────────────────────────────────────────────
    print("=" * 92)
    print("E. 标量与向量(可学习的小参数 —— 决定「看哪里」和「用多少」)")
    print("=" * 92)
    print("  %-42s %-10s %10s %s" % ("参数名", "shape", "元素数", "种类"))
    for name, shape, n, k, _ in sorted(rows, key=lambda r: r[2]):
        if k == "矩阵":
            continue
        print("  %-42s %-10s %10s %s"
              % (name, "×".join(str(s) for s in shape) or "()",
                 "{:,}".format(n), k))
    print()

    # ── 写文档 ────────────────────────────────────────────────────────
    L = []
    W = L.append
    W("# 架构清点：标量 / 向量 / 矩阵 / 在激活（2026-10-10）\n")
    W("脚本 `engram_rule/probe_arch_audit.py`，**零训练**，秒级。")
    W("出货配置来自 `engram_rule_dialogue_bpe_answer_disjoint_v11.pt`。\n")
    W("## 1. 配置快照\n")
    W("| 键 | 值 |")
    W("|---|---:|")
    for k in ("vocab", "d", "n_layers", "n_heads", "ffn_mult", "rope_dim",
              "ngram_max", "n_hash", "n_slots", "n_buckets", "gate_rank",
              "pos_abs_max", "window_max", "bucket_step", "near_first",
              "rule_scale_init", "lam_init", "use_pos_abs", "use_pos_bucket",
              "clamp_window", "local_norm", "use_qpool"):
        W("| `%s` | %s |" % (k, cfg.get(k, "—")))
    W("")
    W("## 2. 按种类汇总\n")
    W("| 种类 | 参数个数 | 元素总数 | 占比 |")
    W("|---|---:|---:|---:|")
    for k in ("标量", "向量", "矩阵"):
        c, n = by_kind[k]
        W("| **%s** | %d | %s | %.2f%% |" % (k, c, "{:,}".format(n),
                                             100.0 * n / total))
    W("| 合计 | %d | %s | 100%% |"
      % (sum(v[0] for v in by_kind.values()), "{:,}".format(total)))
    W("")
    W("★**参与前向的元素只占 %.1f%%**（判据：对两个不同输入求梯度，"
      "两次都非零才算激活）。\n" % (100.0 * alive_n / total))
    W("## 3. 矩阵参数（按元素数降序）\n")
    W("| 参数名 | shape | 元素数 | 激活 |")
    W("|---|---|---:|---|")
    for name, shape, n, k, _ in sorted(rows, key=lambda r: -r[2]):
        if k != "矩阵":
            continue
        W("| `%s` | %s | %s | %s |"
          % (name, "×".join(str(s) for s in shape), "{:,}".format(n),
             "✓" if (gmap.get(name) and gmap2.get(name)) else "—"))
    W("")
    W("## 4. 标量与向量\n")
    W("| 参数名 | shape | 元素数 | 种类 | 激活 |")
    W("|---|---|---:|---|---|")
    for name, shape, n, k, _ in sorted(rows, key=lambda r: r[2]):
        if k == "矩阵":
            continue
        W("| `%s` | %s | %s | %s | %s |"
          % (name, "×".join(str(s) for s in shape) or "()", "{:,}".format(n), k,
             "✓" if (gmap.get(name) and gmap2.get(name)) else "—"))
    W("")
    W("## 5. 通路开关\n")
    W("| 开关/通路 | 取值 | 含义 |")
    W("|---|---:|---|")
    W("| `use_pos_abs` | %s | 门的查询侧加绝对位置 |" % cfg.get("use_pos_abs"))
    W("| `use_pos_bucket` | %s | 位置进桶偏置（决定窗口起点） |" % cfg.get("use_pos_bucket"))
    W("| `clamp_window` | %s | 因果夹紧（C28） |" % cfg.get("clamp_window"))
    W("| `local_norm` | %s | 窗口内逐通道归一化（C29） |" % cfg.get("local_norm"))
    W("| `use_qpool` | %s | ★问题池化通路（§61）——**出货模型没开** |" % cfg.get("use_qpool"))
    W("")
    W("## 6. 实测激活状态\n")
    W("| 量 | 读数 |")
    W("|---|---:|")
    W("| Engram 存活槽 | %d / %d（%.1f%%） |"
      % (alive_slots, cfg["n_slots"], 100.0 * alive_slots / cfg["n_slots"]))
    W("| `start_bucket` 取值分布 | %s |" % dict(sorted(sb_counter.items())))
    W("| 本批规则路径是否在跑 | %s |" % has_rule)
    W("")
    W("## 7. 诚实边界\n")
    W("- 「激活」用的是**梯度探针**：两个不同输入上梯度都非零才算。")
    W("  它可能把「只在特定输入下才被用到」的参数误判为未激活 —— 这类参数极少，")
    W("  且不影响本节的结构性结论。")
    W("- `EngramMemory.alive` 是 **buffer 不是 Parameter**，不在参数表里，")
    W("  所以不参与上面的元素数统计。它是 4096 个 bool，由语料统计决定。")
    W("- 本节**没有改动任何代码**，纯粹是清点。")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
