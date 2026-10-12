"""修正版: 只留出名词、保留动词(答案可输出) ⇒ 直接测"一致规则能否迁移到未见主语"。"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from baseline.dense import DenseLM
from corpus_agreement import corpus, T, VOCAB, V_SG, V_PL
from exp_agreement import run as train_run

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def acc_by(m, x, vp, at, hs):
    m.eval()
    V1 = torch.tensor([T[v] for v in V_SG], device=DEV)
    V2 = torch.tensor([T[v] for v in V_PL], device=DEV)
    with torch.no_grad():
        xd = x.to(DEV)
        lg = m(xd)
    p = lg[torch.arange(x.shape[0]), vp].softmax(-1)
    corr = torch.where(hs.to(DEV), p[:, V1].sum(-1) > p[:, V2].sum(-1),
                       p[:, V2].sum(-1) > p[:, V1].sum(-1))
    at_d = at.to(DEV)
    m.train()
    return (float(corr.float().mean()),
            float(corr[~at_d].float().mean()), float(corr[at_d].float().mean()))


def main():
    L = 24
    cfg = RCfg(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=16, gate_rank=16)
    xtr, vtr, atr, htr = corpus(2400, L, seed=1)
    xh, vh, ah, hh = corpus(500, L, seed=3, hold_nouns=True, hold_verbs=False,
                            avoid=set(map(tuple, xtr.tolist())))
    print("设备 %s | ★只留出名词(goat/pig 族), 动词全部见过 ⇒ 答案可输出" % DEV)
    print("  留出集: 未见名词 %d 行, 吸引子占比 %.3f\n" % (xh.shape[0], float(ah.float().mean())))
    print("  %-5s %8s %8s %8s %9s" % ("模型", "总体", "无吸引", "吸引子", "干扰"))
    for tag in ("ER", "稠密"):
        rs = {"a": [], "n": [], "t": []}
        for sd in range(3):
            torch.manual_seed(sd)
            m = EngramRuleLM(cfg).to(DEV) if tag == "ER" else DenseLM(cfg).to(DEV)
            if tag == "ER":
                m.engram.build_from_corpus(xtr.reshape(1, -1)[:, :50000])
            train_run(m, xtr, vtr, htr, 400, 256, 3e-3, sd)
            a, n, t = acc_by(m, xh, vh, ah, hh)
            rs["a"].append(a); rs["n"].append(n); rs["t"].append(t)
        print("  %-5s %8.4f %8.4f %8.4f %9.4f   (3种子, 极差 %.4f)"
              % (tag, sum(rs["a"]) / 3, sum(rs["n"]) / 3, sum(rs["t"]) / 3,
                 sum(rs["n"]) / 3 - sum(rs["t"]) / 3, max(rs["a"]) - min(rs["a"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
