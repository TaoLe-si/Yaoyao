"""★ 规则层"能否把窗口放到正确距离"实验 (SPEC §10 路线① 第二子项) + 两个代码级修复的对照。

§25 判定: 查询位 q=vp−1, 主语位 1, 因果回看偏移 d=q−1=vp−2=3k。旧实验把 d 写成 1+3k 且用 q<W 分类, 两处各差一位;
旧结论里"窗内/超窗"分组和覆盖率均不成立, 需要按同一干净口径重跑。
本轮逐项对照:
  ① C28 因果窗口夹紧: 起点偏移夹到 [max(0,q−W+1), q], 让长度 W 的窗口完整落在因果前缀内 (q<W−1 时只含现有 token)。
     旧行为允许整窗 j<0 ⇒ 窗口分支无有效 K/V; 窗口分支梯度为零 (§4.4 direct 是独立旁路)。
  ② 路线①(b) 位置→桶偏置: 位置(可数统计)进桶偏置 ⇒ 起点能随"数了多少步"移动 (零初值 ⇒ 初始恒等)。
  ③ C29 分母掩码: 局部归一化分母只统计有效位置。
窗口 [0,W−1], 超窗严格为 d≥W; 4 层堆叠的覆盖率按**任一层覆盖**统计, 同时报第0层覆盖。
干净口径: 查询位=vp−1 (C24); near_first=0。消融分整体规则和仅 §4.4 direct, 判断窗口分支是否必要。
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from baseline.dense import DenseLM
from corpus_agreement import corpus, make_sentence, T, VOCAB, V_SG, V_PL, PREP, N_SG, N_PL, END

DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = 16
L = 40
NSEED = 3


def train(m, xtr, vtr, htr, steps, bs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    m.train()
    for _ in range(steps):
        i = torch.randint(0, xtr.shape[0], (bs,), generator=g)
        x = xtr[i].to(DEV)
        lg = m(x)
        q = vtr[i] - 1
        loss = F.cross_entropy(lg[torch.arange(bs), q], x[torch.arange(bs), vtr[i]])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    return m


@torch.no_grad()
def _acc(m, x, vp, hs, inw):
    """窗内/超窗单复数准确率 (查询位 = vp-1)。"""
    lg = m(x.to(DEV))
    V1 = torch.tensor([T[v] for v in V_SG], device=DEV)
    V2 = torch.tensor([T[v] for v in V_PL], device=DEV)
    q = (vp - 1).clamp(min=0)
    p = lg[torch.arange(x.shape[0]), q.to(DEV)].softmax(-1)
    corr = torch.where(hs.to(DEV), p[:, V1].sum(-1) > p[:, V2].sum(-1),
                       p[:, V2].sum(-1) > p[:, V1].sum(-1)).cpu()
    return (float(corr[inw].float().mean()), float(corr[~inw].float().mean()))


@torch.no_grad()
def evaluate(m, x, vp, hs, need, inw):
    """窗口偏移 off 是相对回看距离 q-j, 与因果距离/C28 的定义直接一致。"""
    m.eval()
    ab = [b.attn for b in m.blocks if hasattr(b, "attn")]
    for a in ab:
        a.record = True
    res = _acc(m, x, vp, hs, inw)
    # ★ 诊断必须在**消融之前**读: ablate_rule=True 会把 bias 清零 ⇒ startb 归 0 ⇒ 读数被污染
    st = cov = per_k = per_raw = None
    if ab and getattr(ab[0], "last_window_off", None) is not None:
        a0 = ab[0]
        q = (vp - 1).clamp(min=0)
        so = a0.last_window_off[torch.arange(x.shape[0]), q.to(DEV), 0].float().cpu()
        raw = a0.last_startb[torch.arange(x.shape[0]), q.to(DEV)].float().cpu()
        st = (float(so[inw].mean()), float(so[~inw].mean()))
        # ★ 覆盖率必须**看任一层**: 预测由 4 层堆叠产出, 只看第 0 层会把"深层盖住了"读成没盖住
        #   (§26 实测: 第0层覆盖 0.58 而 acc 0.999 ⇒ 旧读数与结果矛盾, 是**指标错**不是机制错)
        covs = []
        for a in ab:
            if getattr(a, "last_window_off", None) is None:
                continue
            sa = a.last_window_off[torch.arange(x.shape[0]), q.to(DEV), 0].float().cpu()
            covs.append((sa <= need) & (need <= sa + (W - 1)))
        c_any = torch.stack(covs).any(0).float()
        c0 = (float(covs[0][inw].float().mean()), float(covs[0][~inw].float().mean()))
        cov = (float(c_any[inw].mean()), float(c_any[~inw].mean()), c0[0], c0[1])   # (任一层内/超, 第0层内/超)
        kvec = (need // 3)
        per_k = [float(so[kvec == k].mean()) if bool((kvec == k).any()) else float("nan")
                 for k in range(1, 11)]
        per_raw = [float(raw[kvec == k].mean()) if bool((kvec == k).any()) else float("nan")
                   for k in range(1, 11)]
    only_direct = only_window = neither = None
    if ab:
        for a in ab:
            a.ablate_window = True       # 只关窗口聚合, direct 保留
        only_direct = _acc(m, x, vp, hs, inw)
        for a in ab:
            a.ablate_window = False
            a.ablate_direct = True       # 只关 direct, 窗口聚合保留
        only_window = _acc(m, x, vp, hs, inw)
        for a in ab:
            a.ablate_window = True
        neither = _acc(m, x, vp, hs, inw)
        for a in ab:
            a.ablate_window = False
            a.ablate_direct = False
    m.train()
    return res, st, cov, per_k, per_raw, only_window, only_direct, neither


def stratified_holdout(L, n_per_k=60):
    """每个可留出深度 k=1..10 各取相同数量; 训练生成器随后显式避开这些行。
    k=0 的 24 种句子会被训练集穷尽, 因此不伪称它可留出; 报告严格限制在实际可留出的 k。"""
    seen = set()
    rows, vps, ats, hs = [], [], [], []
    for k in range(1, 11):
        g = torch.Generator().manual_seed(3000 + k)
        budget = n_per_k * 100
        count = 0
        while count < n_per_k and budget > 0:
            budget -= 1
            toks, vp, at, h = make_sentence(g, max_pp=10)
            if (vp - 2) // 3 != k or len(toks) > L:
                continue
            row = tuple(toks + [T[END]] * (L - len(toks)))
            if row in seen:
                continue
            seen.add(row)
            rows.append(list(row)); vps.append(vp); ats.append(at); hs.append(h)
            count += 1
        assert count == n_per_k, "k=%d 分层留出只生成 %d/%d" % (k, count, n_per_k)
    x = torch.tensor(rows, dtype=torch.long)
    return x, torch.tensor(vps), torch.tensor(ats, dtype=torch.bool), torch.tensor(hs, dtype=torch.bool)


def main():
    cfg = dict(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=W,
               gate_rank=16, near_first=0.0)          # ★ 初值不焊死起点
    xh, vh, ah, hh = stratified_holdout(L, n_per_k=60)
    xtr, vtr, atr, htr = corpus(2400, L, seed=1, max_pp=10, avoid=set(map(tuple, xh.tolist())))
    assert not (set(map(tuple, xtr.tolist())) & set(map(tuple, xh.tolist())))
    q = (vh - 1).clamp(min=0)
    need = (q - 1).clamp(min=0)                      # 主语 token 在位 1; 查询位 q ⇒ 回看偏移 q-1=3k
    k = (vh - 2) // 3
    assert bool((vh == 2 + 3 * k).all()) and bool((need == 3 * k).all())
    assert set(k.tolist()) == set(range(1, 11))
    inw = need < W                                   # 窗口偏移 [0,W-1], 含 W-1 边界
    print("设备 %s | k=1..10 分层 | 查询 q=vp−1, 主语偏移 d=q−1=3k ∈ [%d,%d]; 窗口 [0,%d], 超窗(d≥%d) 占 %.3f"
          % (DEV, int(need.min()), int(need.max()), W - 1, W, float((~inw).float().mean())))
    print("训练 %d 行 / 测试 %d 行 | near_first=0 | 干净口径(vp-1) | %d 种子\n"
          % (xtr.shape[0], xh.shape[0], NSEED))
    print("  %-16s %8s %8s %8s   %-14s %-22s %-11s %-11s %-11s"
          % ("臂", "窗内acc", "超窗acc", "差", "起点偏移(内/超)", "覆盖(任一层|第0层)", "仅窗口", "仅direct", "两者都关"))
    for tag, kw in (("ER-旧(§25复现)", dict(clamp_window=False, use_pos_bucket=False)),
                    ("ER+夹紧", dict(clamp_window=True, use_pos_bucket=False)),
                    ("ER+夹紧+位标", dict(clamp_window=True, use_pos_bucket=True)),
                    ("稠密", None)):
        rows = []
        for sd in range(NSEED):
            torch.manual_seed(sd)
            if kw is None:
                m = DenseLM(RCfg(**cfg)).to(DEV)
            else:
                m = EngramRuleLM(RCfg(**cfg, **kw)).to(DEV)
                m.engram.build_from_corpus(xtr.reshape(1, -1)[:, :50000])
            train(m, xtr, vtr, htr, 500, 256, 3e-3, sd)
            res, st, cov, pk, pr, abl_w, abl_d, abl_all = evaluate(m, xh, vh, hh, need, inw)
            rows.append((res, st, cov, pk, pr, abl_w, abl_d, abl_all))
            fmt = lambda z: "—" if z is None else "%.3f/%.3f" % z
            print("      · %s 种子 %d: acc %.4f/%.4f | 仅窗口 %s | 仅direct %s | 两者都关 %s"
                  % (tag, sd, res[0], res[1], fmt(abl_w), fmt(abl_d), fmt(abl_all)))
            sys.stdout.flush()
        acc = [sum(r[0][i] for r in rows) / NSEED for i in (0, 1)]
        fs = lambda: ("%.2f / %.2f" % (sum(r[1][0] for r in rows) / NSEED,
                                       sum(r[1][1] for r in rows) / NSEED)) if rows[0][1] else "—"
        cs = lambda: ("%.2f/%.2f | %.2f/%.2f" % (sum(r[2][0] for r in rows) / NSEED,
                                                 sum(r[2][1] for r in rows) / NSEED,
                                                 sum(r[2][2] for r in rows) / NSEED,
                                                 sum(r[2][3] for r in rows) / NSEED)) if rows[0][2] else "—"
        abw_ = lambda: ("%.3f/%.3f" % (sum(r[5][0] for r in rows) / NSEED,
                                        sum(r[5][1] for r in rows) / NSEED)) if rows[0][5] else "—"
        abd_ = lambda: ("%.3f/%.3f" % (sum(r[6][0] for r in rows) / NSEED,
                                        sum(r[6][1] for r in rows) / NSEED)) if rows[0][6] else "—"
        aba_ = lambda: ("%.3f/%.3f" % (sum(r[7][0] for r in rows) / NSEED,
                                        sum(r[7][1] for r in rows) / NSEED)) if rows[0][7] else "—"
        print("  %-16s %8.4f %8.4f %8.4f   %-14s %-16s %-11s %-11s %-11s"
              % (tag, acc[0], acc[1], acc[0] - acc[1], fs(), cs(), abw_(), abd_(), aba_()))
        if rows[0][3] is not None:
            pk = [sum(r[3][k] for r in rows) / NSEED for k in range(10)]
            pr = [sum(r[4][k] for r in rows) / NSEED for k in range(10)]
            print("      各 k 实际偏移(夹紧后, k=1..10): " + " ".join("%.1f" % v for v in pk))
            print("      各 k 起点桶(夹紧前, k=1..10):   " + " ".join("%.1f" % v for v in pr))
        sys.stdout.flush()
    print("  需要偏移(k=1..10):       " + " ".join("%.0f" % (3 * k) for k in range(1, 11)))
    print("  判据: ③ 超窗 acc 明显 > 0.5 且覆盖率同步上升 ⇒ 放置瓶颈=信息通路 (路线①(b)成立);")
    print("        ③ 覆盖率上升但 acc 仍 ≈0.5 ⇒ 瓶颈转移到窗内分辨(gate);")
    print("        ② 窗内 acc 恢复而 ③ 不涨 ⇒ 夹紧修好了退化, 位置通路没被用上;")
    print("        可留出测试深度为 k=1..10; k=0 的句子空间已被训练集穷尽, 不计入泛化比较。")
    print("        分别关窗口、direct、两者都关后 acc 掉到何处, 用来识别窗口分支/直连分支是否必要。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
