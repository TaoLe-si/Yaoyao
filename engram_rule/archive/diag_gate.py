"""门把权重给了谁: 在动词位置比较"给中心词的份额 vs 给吸引子的份额"(只看吸引子句)。
规则层已确认惰性(起点冻结), 判决全在 SwiGLU 门。判据可失败性由 C23 保证(last_contrib 随输入变)。"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from corpus_agreement import corpus, T, VOCAB, V_SG, V_PL

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def head_offset(x, vp):
    """中心词到动词的偏移 = 1+3k (k = 动词前 'the' 的个数 - 1)"""
    out = []
    for i in range(x.shape[0]):
        v = int(vp[i])
        n_the = sum(1 for j in range(v) if int(x[i, j]) == T["the"])
        out.append(1 + 3 * (n_the - 1))
    return torch.tensor(out)


def main():
    L = 24
    cfg = RCfg(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=16, gate_rank=16)
    xtr, vtr, atr, htr = corpus(2400, L, seed=1)
    xh, vh, ah, hh = corpus(500, L, seed=3, hold_nouns=True, hold_verbs=True,
                            avoid=set(map(tuple, xtr.tolist())))
    dh = head_offset(xh, vh)
    print("设备 %s | 只看吸引子句 (最近名词=吸引子, 中心词在偏移 1+3k)" % DEV)
    print("  %-5s %8s %12s %12s  %s" % ("种子", "正确率", "中心词份额", "吸引子份额", "门读了谁"))
    for sd in range(3):
        torch.manual_seed(sd)
        m = EngramRuleLM(cfg).to(DEV)
        m.engram.build_from_corpus(xtr.reshape(1, -1)[:, :50000])
        g = torch.Generator().manual_seed(sd)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=400, pct_start=0.1)
        for _ in range(400):
            i = torch.randint(0, xtr.shape[0], (256,), generator=g)
            x = xtr[i].to(DEV)
            lg = m(x)
            loss = F.cross_entropy(lg[torch.arange(256), vtr[i]], x[torch.arange(256), vtr[i]])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        m.eval(); a = m.blocks[0].attn; a.record = True
        with torch.no_grad():
            xd = xh[:128].to(DEV)
            lg = m(xd)
        contrib = a.last_contrib                                   # (B,L,W)
        rows = (torch.arange(128), vh[:128])
        w = contrib[rows].abs() + 1e-9
        w = w / w.sum(-1, keepdim=True)
        at = ah[:128]
        ho, ao = dh[:128], torch.ones(128, dtype=torch.long)
        hs = float(w[torch.arange(128), ho][at].mean())
        as_ = float(w[torch.arange(128), ao][at].mean())
        V1 = torch.tensor([T[v] for v in V_SG], device=DEV)
        V2 = torch.tensor([T[v] for v in V_PL], device=DEV)
        with torch.no_grad():
            p = lg[torch.arange(128), vh[:128]].softmax(-1)
            corr = torch.where(hh[:128].to(DEV), p[:, V1].sum(-1) > p[:, V2].sum(-1),
                               p[:, V2].sum(-1) > p[:, V1].sum(-1))
            acc = float(corr.float().mean())
        print("  %-5d %8.4f %12.4f %12.4f  %s" % (sd, acc, hs, as_,
              "中心词" if hs > as_ else "**吸引子**"))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
