"""定位 C13 (同种子两次训练逐位一致) 被我本轮哪一处改动破坏。
跑法: python engram_rule/diag_det.py
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM

VOCAB, B, L = 256, 4, 16
g0 = torch.Generator().manual_seed(1)
DTR = torch.randint(0, VOCAB, (64, L), generator=g0)


def two_steps(kw):
    c = RCfg(vocab=VOCAB, d=64, n_layers=2, n_heads=4, ffn_mult=2, rope_dim=16,
             ngram_max=3, n_hash=3, n_slots=8192, n_buckets=8, **kw)
    torch.manual_seed(7)
    m = EngramRuleLM(c)
    m.engram.alive[:] = True
    g = torch.Generator().manual_seed(7)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(2):
        i = torch.randint(0, DTR.shape[0], (4,), generator=g)
        x = DTR[i]
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, c.vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    h = torch.zeros(1)
    for p in m.parameters():
        h = h + float(p.detach().double().sum())
    return float(h)


for tag, kw in (("默认(三处改动全开)", {}),
                ("关 pos_bucket", dict(use_pos_bucket=False)),
                ("关夹紧", dict(clamp_window=False)),
                ("两开关全关(只留分母掩码)", dict(use_pos_bucket=False, clamp_window=False))):
    ds = [abs(two_steps(kw) - two_steps(kw)) for _ in range(3)]
    print("  %-22s 三次复跑的指纹差: %s" % (tag, " ".join("%.3e" % d for d in ds)))
    sys.stdout.flush()
