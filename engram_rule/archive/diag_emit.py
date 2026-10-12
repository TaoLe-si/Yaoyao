"""0.02 档在动词位置到底输出什么? (判清: 是学不会用未见 token, 还是系统性反向)
同一次训练后, 打印留出词项语料动词位置的 top-5 预测词 (解码回字符串)。"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from baseline.dense import DenseLM
from corpus_agreement import corpus, TOKS, T, VOCAB, V_SG, V_PL

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    L = 24
    cfg = RCfg(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=16, gate_rank=16)
    xtr, vtr, _, htr = corpus(2400, L, seed=1)
    xh, vh, ah, hh = corpus(500, L, seed=3, hold_nouns=True, hold_verbs=True,
                            avoid=set(map(tuple, xtr.tolist())))
    for tag, mk in (("ER", lambda: EngramRuleLM(cfg).to(DEV)), ("稠密", lambda: DenseLM(cfg).to(DEV))):
        torch.manual_seed(0)                      # 种子 0 = 已知的 0.02 档
        m = mk()
        if tag == "ER":
            m.engram.build_from_corpus(xtr.reshape(1, -1)[:, :50000])
        g = torch.Generator().manual_seed(0)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=400, pct_start=0.1)
        for _ in range(400):
            i = torch.randint(0, xtr.shape[0], (256,), generator=g)
            x = xtr[i].to(DEV)
            lg = m(x)
            loss = F.cross_entropy(lg[torch.arange(256), vtr[i]], x[torch.arange(256), vtr[i]])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            xd = xh[:8].to(DEV)
            lg = m(xd)
        print("\n===== %s (种子 0) 在留出词项句子的动词位置输出 top-5 =====" % tag)
        print("  (真值动词 ∈ {likes, like} — 训练里从未出现过这两个词)")
        for r in range(8):
            top = lg[r, int(vh[r])].topk(5)
            words = ["%s(%.2f)" % (TOKS[int(j)], float(p)) for p, j in zip(top.values, top.indices)]
            truth_sg = "单数" if bool(hh[r]) else "复数"
            print("  行%d 真值: %s  ⇒  %s" % (r, truth_sg, "  ".join(words)))
        # 统计: 预测里"留出动词 likes/like"占了多少概率
        with torch.no_grad():
            lgall = m(xh.to(DEV))[torch.arange(500), vh].softmax(-1)
        p_like = float(lgall[:, T["likes"]].mean() + lgall[:, T["like"]].mean())
        p_seen = float(lgall[:, [T[v] for v in ("sees", "see", "chases", "chase")]].sum(-1).mean())
        print("  ⇒ 留出动词(likes/like)平均概率 %.4f | 见过动词(sees/chases…)平均概率 %.4f"
              % (p_like, p_seen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
