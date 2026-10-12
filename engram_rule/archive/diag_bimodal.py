"""双峰的优化层诊断: 成功种子 vs 失败种子**在哪里分叉**, 分叉点上窗口起点是否盖住中心词。

不猜原因, 只观测三样随时间的变化:
  A 留出词项正确率     —— 双峰体现在哪一步分开
  B 动词位置的窗口起点 mean(startb)  —— 规则学出来的那个数
  C 窗口覆盖中心词的比例 —— 中心词距离 = 1+3k (k = PP 数)
"""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from corpus_agreement import corpus, T, VOCAB, V_SG, V_PL
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def vpos_info(x, vpos):
    """每个动词位置的中心词距离 = 1+3k (k = 动词前有几个 PP)"""
    d = []
    for i in range(x.shape[0]):
        vp = int(vpos[i])
        # 计 PP 数: 动词前 "the" 的个数 - 1
        n_the = sum(1 for j in range(vp) if int(x[i, j]) == T["the"])
        d.append(1 + 3 * (n_the - 1))
    return torch.tensor(d)


@torch.no_grad()
def probe(m, x, vpos, head_sg, dists):
    m.eval()
    xd = x.to(DEV)
    lg = m(xd)
    V1 = torch.tensor([T[v] for v in V_SG], device=DEV)
    V2 = torch.tensor([T[v] for v in V_PL], device=DEV)
    p = lg[torch.arange(x.shape[0]), vpos].softmax(-1)
    corr = torch.where(head_sg.to(DEV), p[:, V1].sum(-1) > p[:, V2].sum(-1),
                       p[:, V2].sum(-1) > p[:, V1].sum(-1))
    # 窗口起点: 重新算一次 bias (需要一次带 record 的前向)
    a = m.blocks[0].attn
    a.record = True
    m(xd)
    start = a.last_startb                                 # (B,L)
    sv = start[torch.arange(x.shape[0]), vpos].float()
    cov = (sv <= dists.to(DEV).float()) & (dists.to(DEV).float() < sv + m.c.window_max)
    m.train()
    return float(corr.float().mean()), float(sv.mean()), float(cov.float().mean())


def main():
    L = 24
    cfg = RCfg(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=16, gate_rank=16)
    xtr, vtr, atr, htr = corpus(2400, L, seed=1)
    xh, vh, ah, hh = corpus(500, L, seed=3, hold_nouns=True, hold_verbs=True,
                            avoid=set(map(tuple, xtr.tolist())))
    dh = vpos_info(xh, vh)
    print("设备 %s | 留出词项 %d 行 | 中心词距离分布: %s"
          % (DEV, xh.shape[0], {int(k): int(v) for k, v in zip(*torch.unique(dh, return_counts=True))}))
    print("\n  %-5s %-6s %8s %10s %10s" % ("种子", "步", "留出正确率", "窗口起点", "覆盖中心词"))
    for sd in range(3):
        torch.manual_seed(sd)
        m = EngramRuleLM(cfg).to(DEV)
        m.engram.build_from_corpus(xtr.reshape(1, -1)[:, :50000])
        g = torch.Generator().manual_seed(sd)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=400, pct_start=0.1)
        for s in range(1, 401):
            i = torch.randint(0, xtr.shape[0], (256,), generator=g)
            x = xtr[i].to(DEV)
            lg = m(x)
            loss = F.cross_entropy(lg[torch.arange(256), vtr[i]], x[torch.arange(256), vtr[i]])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            if s % 100 == 0:
                acc, sv, cov = probe(m, xh[:256], vh[:256], hh[:256], dh[:256])
                print("  %-5d %-6d %8.4f %10.2f %10.3f" % (sd, s, acc, sv, cov))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
