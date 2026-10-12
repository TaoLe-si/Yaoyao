"""★ 干净口径的一致任务: 查询位 = 动词前一位 (vp-1), 动词不在自己的注意力窗内 ⇒ 抄写捷径不存在。
训练目标仍是动词 token, 但从**前缀**预测。3 种子 × 2 模型 × {同分布, 留出名词}。"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from baseline.dense import DenseLM
from corpus_agreement import corpus, T, VOCAB, V_SG, V_PL

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train(m, xtr, vtr, htr, steps, bs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    m.train()
    for _ in range(steps):
        i = torch.randint(0, xtr.shape[0], (bs,), generator=g)
        x = xtr[i].to(DEV)
        lg = m(x)
        q = vtr[i] - 1                                   # ★ 查询位 = 动词前一位
        tgt = x[torch.arange(bs), vtr[i]]                # 目标 = 动词 token
        loss = F.cross_entropy(lg[torch.arange(bs), q], tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    return m


@torch.no_grad()
def acc(m, x, vp, at, hs):
    m.eval()
    V1 = torch.tensor([T[v] for v in V_SG], device=DEV)
    V2 = torch.tensor([T[v] for v in V_PL], device=DEV)
    lg = m(x.to(DEV))
    q = (vp - 1).clamp(min=0)
    p = lg[torch.arange(x.shape[0]), q].softmax(-1)
    corr = torch.where(hs.to(DEV), p[:, V1].sum(-1) > p[:, V2].sum(-1),
                       p[:, V2].sum(-1) > p[:, V1].sum(-1))
    at_d = at.to(DEV)
    m.train()
    return (float(corr.float().mean()), float(corr[~at_d].float().mean()),
            float(corr[at_d].float().mean()))


def main():
    L = 24
    cfg = RCfg(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=16, gate_rank=16)
    xtr, vtr, atr, htr = corpus(2400, L, seed=1)
    xin, vin, ain, hin = corpus(400, L, seed=2, avoid=set(map(tuple, xtr.tolist())))
    xh, vh, ah, hh = corpus(400, L, seed=3, hold_nouns=True, hold_verbs=False,
                            avoid=set(map(tuple, xtr.tolist())) | set(map(tuple, xin.tolist())))
    print("设备 %s | ★干净口径: 查询位 = 动词前一位 (抄写捷径不存在) | 400 步 × 3 种子" % DEV)
    print("  %-5s %-10s %8s %8s %8s %9s" % ("模型", "集合", "总体", "无吸引", "吸引子", "干扰"))
    for tag, mk in (("ER", lambda: EngramRuleLM(cfg).to(DEV)),
                    ("稠密", lambda: DenseLM(cfg).to(DEV))):
        for name, (x, vp, at, hs) in (("同分布", (xin, vin, ain, hin)), ("留出名词", (xh, vh, ah, hh))):
            rs = {"a": [], "n": [], "t": []}
            for sd in range(3):
                torch.manual_seed(sd)
                m = mk()
                if tag == "ER":
                    m.engram.build_from_corpus(xtr.reshape(1, -1)[:, :50000])
                train(m, xtr, vtr, htr, 400, 256, 3e-3, sd)
                a, n, t = acc(m, x, vp, at, hs)
                rs["a"].append(a); rs["n"].append(n); rs["t"].append(t)
            print("  %-5s %-10s %8.4f %8.4f %8.4f %9.4f  (极差 %.4f)"
                  % (tag, name, sum(rs["a"]) / 3, sum(rs["n"]) / 3, sum(rs["t"]) / 3,
                     sum(rs["n"]) / 3 - sum(rs["t"]) / 3, max(rs["a"]) - min(rs["a"])))
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
