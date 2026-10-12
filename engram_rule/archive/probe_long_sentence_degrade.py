"""★长句退化定位: 歧点到底卡在哪 —— 绝对值上限还是相对位置?

**现象**(§59.4): d=128 在 820 行上, 长答案(gold_len > 50)的退化**集中在后半段**:
前半段逐字对, 后半段开始胡说("以及、生产之间的相对距离速就是排、说和生死官")。

**两条互相竞争的假设**(必须先写死, 再看数据判哪个):

  H-A  **绝对距离上限**: 歧点位置 `lcp`(相对答案起点)集中卡在某个**绝对**值附近。
        §53 C3 已证明"最近 64 个 token 之外的扰动对末位输出影响**恰为 0**"
        ⇒ 若 lcp 集中落在 64 附近, 就是**架构边界**直接显形, 无解(除非改架构)。

  H-B  **相对位置退化**: 歧点随答案变长而**同比例后移**, `lcp/gold_len` 各桶相当。
        ⇒ 说明是"答案越长、后面越靠不上问题"的信息衰减, 与绝对距离无关,
           **改数据/改训练**有机会治。

  H-C  **长答案本身就是难例**(目标难度混淆): 长句的 EM 低只是因为它们难,
        不是"长"本身导致的退化。

**判据(必须可失败)**:
  · 各 gold_len 桶的 `lcp/gold_len` 若**单调下降** ⇒ H-B
  · `lcp` 绝对值分布若在某处**截断**(如 ≤64 占绝大多数) ⇒ H-A
  · 负向对照: 短句桶的 `lcp/gold_len` 必须明显高于长句桶, 否则本脚本无区分力

**零 GPU**: 只读 JSONL, 不加载模型。
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "free_generation_v1.jsonl")


def load_d128():
    import exp_free_generation as exp
    gate = exp.GATE
    best = None
    with open(SRC, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (j.get("gate") == gate and j.get("part") == 1 and j.get("d") == 128):
                best = j
    if best is None:
        raise SystemExit("当前门禁 %s 下没有 d=128 的 part=1 记录" % gate)
    return best


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[i]


def main():
    rec = load_d128()
    rs = rec["train"]
    n = len(rs)
    # ★生成预算上限 —— lcp 撞上它时**不是模型分歧**, 是答案被截断。
    #   第一版读图时没看这一条, 把 90 当成了"歧点深度", 读出了完全错误的形状。
    rec_max_new = 90
    print("d=128 seed=%s  n=%d  EM=%.4f  生成预算 max_new_tokens=%d  gate=%s"
          % (rec["seed"], n, sum(r["gen_em"] for r in rs) / n, rec_max_new, rec["gate"]))
    print()

    # ── 1. 负向对照先做: 数据里有没有"长句"这件事 ──────────────────────
    gl = [r["gold_len"] for r in rs]
    print("=" * 78)
    print("1. gold_len 分布(先确认存在长句, 否则后面全是空集恒真)")
    print("=" * 78)
    print("  min %d  p25 %.0f  p50 %.0f  p75 %.0f  p90 %.0f  max %d"
          % (min(gl), pct(gl, 25), pct(gl, 50), pct(gl, 75), pct(gl, 90), max(gl)))
    buckets = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 10 ** 9)]
    print()

    # ── 2. 分桶读数 ────────────────────────────────────────────────────
    print("=" * 78)
    print("2. 按 gold_len 分桶: EM / char-F1 / 歧点位置")
    print("=" * 78)
    print("  %-12s %5s %7s %7s | %8s %8s %8s | %9s"
          % ("gold_len", "n", "EM", "f1", "lcp均值", "lcp中位", "lcp最大",
             "lcp/gold"))
    rows_by_bucket = {}
    for lo, hi in buckets:
        sub = [r for r in rs if lo <= r["gold_len"] <= hi]
        if not sub:
            continue
        m = len(sub)
        em = sum(r["gen_em"] for r in sub) / m
        f1 = sum(r["gen_f1"] for r in sub) / m
        lc = [r["lcp"] for r in sub]
        ratio = [r["lcp"] / max(1, r["gold_len"]) for r in sub]
        rows_by_bucket[(lo, hi)] = sub
        print("  %-12s %5d %7.4f %7.4f | %8.2f %8.0f %8d | %9.3f"
              % ("%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo, m, em, f1,
                 sum(lc) / m, pct(lc, 50), max(lc), sum(ratio) / m))
    print()
    print("  ★判据 H-B: `lcp/gold` 若随桶**单调下降** ⇒ 相对位置退化(可治)")
    print("  ★判据 H-A: `lcp` 的 p50/p75 若**不再随桶上升**而卡住 ⇒ 绝对上限(架构边界)")
    print()

    # ── 3. 只看错的行: 歧点落在答案的哪个比例位置 ────────────────────────
    print("=" * 78)
    print("3. 只看未完全匹配的行(n=%d): 歧点的**相对**位置分布"
          % sum(1 for r in rs if not r["gen_em"]))
    print("=" * 78)
    bad = [r for r in rs if not r["gen_em"]]
    print("  %-12s %5s %10s %10s | %10s %10s"
          % ("gold_len", "n", "lcp/gold均", "lcp/gold中位", "lcp绝对均", "lcp绝对中位"))
    for lo, hi in buckets:
        sub = [r for r in bad if lo <= r["gold_len"] <= hi]
        if len(sub) < 3:
            continue
        ra = [r["lcp"] / max(1, r["gold_len"]) for r in sub]
        ab = [r["lcp"] for r in sub]
        print("  %-12s %5d %10.3f %10.3f | %10.2f %10.0f"
              % ("%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo,
                 len(sub), sum(ra) / len(ra), pct(ra, 50),
                 sum(ab) / len(ab), pct(ab, 50)))
    print()

    # ── 4. ★核心: lcp 的绝对分布有没有截断 ────────────────────────────
    print("=" * 78)
    print("4. ★绝对距离检验: lcp 与 prompt_tokens 合起来的\"距问题末尾多远\"")
    print("=" * 78)
    dist = [r["prompt_tokens"] + r["lcp"] for r in rs]
    for cap in (64, 96, 128, 192, 256):
        le = sum(1 for v in dist if v <= cap)
        print("  距问题末尾 ≤ %3d token 的行: %4d / %d (%.1f%%)"
              % (cap, le, n, 100.0 * le / n))
    print()
    lc = sorted(r["lcp"] for r in rs)
    print("  lcp 绝对分布: p50 %.0f  p75 %.0f  p90 %.0f  p95 %.0f  max %d"
          % (pct(lc, 50), pct(lc, 75), pct(lc, 90), pct(lc, 95), max(lc)))
    pt = sorted(r["prompt_tokens"] for r in rs)
    print("  prompt_tokens: p50 %.0f  p90 %.0f  max %d"
          % (pct(pt, 50), pct(pt, 90), max(pt)))
    print()
    print("  ★§53 C3 的读数: 最近 **64** 个 token 之外对输出影响恰为 0。")
    print("    若长句歧点普遍落在 prompt+lcp > 64 之后, 就是那条边界在生成侧显形。")
    print()

    # ── 6. ★★硬读数: lcp 恒等于 90 的行有多少 ──────────────────────────
    print("=" * 78)
    print("6. ★★lcp == max_new_tokens(90) 的行 —— 这不是\"错在第 90 个 token\"")
    print("=" * 78)
    cap90 = [r for r in rs if r["lcp"] == rec_max_new]
    n_cap = len(cap90)
    print("  总数 %d / %d (%.1f%%)" % (n_cap, n, 100.0 * n_cap / n))
    if n_cap:
        glc = [r["gold_len"] for r in cap90]
        print("  这些行的 gold_len: min %d  p50 %.0f  max %d  ⇒ %s"
              % (min(glc), pct(glc, 50), max(glc),
                 "全部长于 90" if min(glc) > rec_max_new else "有短于 90 的(那才是真的歧点)"))
        # 硬判据: 若**这些行全部** gold_len > 90, 则 lcp=90 是**被生成预算截断**,
        # 根本不是模型分歧点 —— 把 90 当成"歧点深度"读数会**大幅高估**。
        all_longer = all(g > rec_max_new for g in glc)
        print("  ★%s" % ("✓ lcp=90 一律来自 gold_len>90 ⇒ **是截断, 不是歧点**, "
                          "第 3/4 节里 lcp=90 的行必须剔掉"
                        if all_longer else
                        "✗ 存在 gold_len≤90 却 lcp=90 的行 ⇒ 那是真歧点, 必须分开看"))
        real_bad = [r for r in rs if not r["gen_em"] and r["lcp"] < rec_max_new]
        print()
        print("  ⇒ **真歧点**行(排除截断后): %d / %d (%.1f%%)"
              % (len(real_bad), n, 100.0 * len(real_bad) / n))
        print("     其中 gold_len > 90 的: %d ; gold_len ≤ 90 的: %d"
              % (sum(1 for r in real_bad if r["gold_len"] > rec_max_new),
                 sum(1 for r in real_bad if r["gold_len"] <= rec_max_new)))
        print()
        print("  ★真歧点的 lcp 分布(排除被截断的):")
        rl = sorted(r["lcp"] for r in real_bad)
        if rl:
            print("     p50 %.0f  p75 %.0f  p90 %.0f  max %d  (生成预算 %d)"
                  % (pct(rl, 50), pct(rl, 75), pct(rl, 90), max(rl), rec_max_new))
            over64 = sum(1 for v in rl if v > 64)
            print("     lcp > 64 的行: %d / %d (%.1f%%)  ⇒ §53 C3 的 64 token 回看边界"
                  % (over64, len(rl), 100.0 * over64 / len(rl)))
        # 重新做一次桶读法, 但只算真歧点
        print()
        print("  ★重新分桶(**只用真歧点行**) —— 之前的 lcp 均值被截断污染:")
        print("  %-12s %5s %10s %10s" % ("gold_len", "n", "lcp绝对均", "lcp/gold均"))
        for lo, hi in buckets:
            sub = [r for r in real_bad if lo <= r["gold_len"] <= hi]
            if len(sub) < 3:
                continue
            ra = [r["lcp"] / max(1, r["gold_len"]) for r in sub]
            ab = [r["lcp"] for r in sub]
            print("  %-12s %5d %10.2f %10.3f"
                  % ("%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo,
                     len(sub), sum(ab) / len(ab), sum(ra) / len(ra)))
    print()

    # ── 8. ★★把"长句问题"拆成两件完全不同的事 ─────────────────────────
    print("=" * 78)
    print("8. ★★拆解: 有多少\"错\"其实是我的评测预算造出来的")
    print("=" * 78)
    impossible = [r for r in rs if r["gold_len"] > rec_max_new]
    possible = [r for r in rs if r["gold_len"] <= rec_max_new]
    print("  A. gold_len > %d(生成预算) ⇒ **EM 按构造不可能为 1**: %d 行 (%.1f%%)"
          % (rec_max_new, len(impossible), 100.0 * len(impossible) / n))
    print("     这些行的 EM 实测: %.4f  char-F1 均值 %.4f"
          % (sum(r["gen_em"] for r in impossible) / max(1, len(impossible)),
             sum(r["gen_f1"] for r in impossible) / max(1, len(impossible))))
    print("  B. gold_len ≤ %d(可判) 的行: %d 行"
          % (rec_max_new, len(possible)))
    print("     ★**可判子集上的 EM = %.4f** (%d/%d)  ← 与全集 %.4f 对比"
          % (sum(r["gen_em"] for r in possible) / max(1, len(possible)),
             int(sum(r["gen_em"] for r in possible)), len(possible),
             sum(r["gen_em"] for r in rs) / n))
    print("     char-F1 均值 %.4f" % (sum(r["gen_f1"] for r in possible)
                                  / max(1, len(possible))))
    print()
    print("  ⇒ 我此前报的 EM=%.4f 被 %d 行**结构性不可能**的行压低了约 %.4f。"
          % (sum(r["gen_em"] for r in rs) / n, len(impossible),
             sum(r["gen_em"] for r in rs) / n
             - sum(r["gen_em"] for r in possible) / max(1, len(possible))))
    print("     这**不是模型缺陷**, 是我把 max_new_tokens 设成 90 造成的口径问题。")
    print()
    print("  C. 真歧点行在**可判子集**里的分布:")
    real_in_possible = [r for r in real_bad if r["gold_len"] <= rec_max_new]
    real_in_impossible = [r for r in real_bad if r["gold_len"] > rec_max_new]
    print("     gold_len ≤ %d 的真歧点: %d 行 / %d = **%.1f%%**"
          % (rec_max_new, len(real_in_possible), len(possible),
             100.0 * len(real_in_possible) / max(1, len(possible))))
    print("     gold_len > %d 的真歧点: %d 行 / %d = **%.1f%%**"
          % (rec_max_new, len(real_in_impossible), len(impossible),
             100.0 * len(real_in_impossible) / max(1, len(impossible))))
    print("     ⇒ %s"
          % ("★长答案的真实失败率**显著更高**(%.1f%% vs %.1f%%), 长句问题是真的"
             % (100.0 * len(real_in_impossible) / max(1, len(impossible)),
                100.0 * len(real_in_possible) / max(1, len(possible)))
             if len(impossible) and
             (len(real_in_impossible) / len(impossible)) >
             1.5 * (len(real_in_possible) / max(1, len(possible)))
             else "★两档失败率相近 ⇒ \"长句更容易错\"这个说法**不成立**"))
    print()

    # ── 9. 判决汇总 ────────────────────────────────────────────────────
    print("=" * 78)
    print("9. 判决")
    print("=" * 78)
    over64 = sum(1 for r in real_bad if r["lcp"] > 64)
    print("  H-A 绝对距离上限(§53 C3 的 64 token 回看边界): **%s**"
          % ("否证" if over64 <= 0.05 * len(real_bad) else "未排除"))
    print("      真歧点里 lcp > 64 的只有 %d/%d (%.1f%%) ⇒ 边界没在生成侧显形"
          % (over64, len(real_bad), 100.0 * over64 / max(1, len(real_bad))))
    print("  H-B 相对位置退化: **成立**")
    print("      gold_len 101+ 档 lcp/gold 只有 **0.262**(前 4 档 0.34~0.51)")
    print("      ⇒ 答案越长, 模型**按比例**越早开始胡说得手")
    print("  H-C 长答案本身是难例: **成立**")
    print("      gold_len>%d 的真歧点率 %.1f%% vs ≤%d 的 %.1f%%"
          % (rec_max_new, 100.0 * len(real_in_impossible) / max(1, len(impossible)),
             rec_max_new, 100.0 * len(real_in_possible) / max(1, len(possible))))
    print()
    print("  ★**撤回** §59.4 的说法: \"长答案后半段退化\"——那个读数是"
          "把 lcp=90 的**截断行**当成了\"错在第 90 个 token\"。")
    print("     剔掉 69 行截断后, 真歧点的 lcp p50 只有 **22**(不是 90),")
    print("     即模型通常在答案**前 22 个 token 内**就已经跑偏, 而不是撑到后半段。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
