"""结论性实验: 主谓一致 + 吸引子干扰上, ER vs 稠密 (3 种子)。
预登记预测: ① 留出词项上 ER ≥ 稠密; ② ER 的**吸引子干扰量** < 稠密 (机制性证据)。
只训动词位置 (任务本身就是这个判决), 两模型完全同条件。"""
import argparse, os, sys, math, time
import torch
import torch.nn.functional as F
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
from model import RCfg, EngramRuleLM
from baseline.dense import DenseLM
from corpus_agreement import corpus, number_acc, VOCAB, T
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run(m, xtr, vtr, htr, steps, bs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    m.train()
    for _ in range(steps):
        i = torch.randint(0, xtr.shape[0], (bs,), generator=g)
        x = xtr[i].to(DEV)
        lg = m(x)
        tgt = x[torch.arange(bs), vtr[i]].to(DEV)                 # 动词位置的真值 token
        loss = F.cross_entropy(lg[torch.arange(bs), vtr[i]], tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    return m


@torch.no_grad()
def eval_sets(m, sets):
    m.eval(); out = {}
    for name, (x, vp, at, hs) in sets.items():
        x = x.to(DEV)
        lg = m(x)[torch.arange(x.shape[0]), vp]
        ok = number_acc(m(x), x.cpu(), vp, hs) if False else None
        # number_acc 需要 logits 的动词位置; 直接在这里算
        V1 = torch.tensor([T[v] for v in __import__("corpus_agreement").V_SG], device=DEV)
        V2 = torch.tensor([T[v] for v in __import__("corpus_agreement").V_PL], device=DEV)
        p = lg.softmax(-1)
        p_sg, p_pl = p[:, V1].sum(-1), p[:, V2].sum(-1)
        hs_d = hs.to(DEV); at_d = at.to(DEV)
        corr = torch.where(hs_d, p_sg > p_pl, p_pl > p_sg)
        o = dict(acc=float(corr.float().mean()),
                 noattr=float(corr[~at_d].float().mean()) if bool((~at_d).any()) else float("nan"),
                 attr=float(corr[at_d].float().mean()) if bool(at_d.any()) else float("nan"))
        o["interference"] = o["noattr"] - o["attr"]
        out[name] = o
    m.train(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    L = 24
    cfg = RCfg(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=16, gate_rank=16)
    xtr, vtr, atr, htr = corpus(2400, L, seed=1)
    xin, vin, ain, hin = corpus(500, L, seed=2, avoid=set(map(tuple, xtr.tolist())))
    xh, vh, ah, hh = corpus(500, L, seed=3, hold_nouns=True, hold_verbs=True,
                            avoid=set(map(tuple, xtr.tolist())) | set(map(tuple, xin.tolist())))
    print("设备 %s | 一致-吸引子语料: 训练 %d / 同分布 %d / **留出词项 %d**"
          % (DEV, xtr.shape[0], xin.shape[0], xh.shape[0]))
    print("  吸引子占比: 训练 %.3f / 同分布 %.3f / 留出 %.3f\n"
          % (float(atr.float().mean()), float(ain.float().mean()), float(ah.float().mean())))
    sets = {"同分布": (xin, vin, ain, hin), "留出词项": (xh, vh, ah, hh)}
    res = {}
    for tag in ("ER", "稠密"):
        rows = {"同分布": [], "留出词项": []}
        for sd in range(a.seeds):
            torch.manual_seed(sd)
            m = EngramRuleLM(cfg).to(DEV) if tag == "ER" else DenseLM(cfg).to(DEV)
            if tag == "ER":
                m.engram.build_from_corpus(xtr.reshape(1, -1)[:, :50000])
            t0 = time.time()
            run(m, xtr, vtr, htr, a.steps, a.bs, a.lr, sd)
            o = eval_sets(m, sets)
            for k in rows:
                rows[k].append(o[k])
            if sd == 0:
                print("  [%s] 首种子用时 %.0fs, 参数 %.3fM" % (tag, time.time() - t0,
                      sum(p.numel() for p in m.parameters()) / 1e6))
        res[tag] = rows
    print("\n================ 结果 (3 种子均值[极差]) ================")
    for k in ("同分布", "留出词项"):
        for tag in ("ER", "稠密"):
            acc = [o["acc"] for o in res[tag][k]]
            na = [o["noattr"] for o in res[tag][k]]
            at = [o["attr"] for o in res[tag][k]]
            print("  %-6s %-6s 正确率 %.4f[%.4f]  无吸引子 %.4f  吸引子 %.4f  **干扰 %.4f**"
                  % (k, tag, sum(acc) / len(acc), max(acc) - min(acc),
                     sum(na) / len(na), sum(at) / len(at),
                     sum(na) / len(na) - sum(at) / len(at)))
    print("\n  预登记判据:")
    e = sum(o["acc"] for o in res["ER"]["留出词项"]) / a.seeds
    d = sum(o["acc"] for o in res["稠密"]["留出词项"]) / a.seeds
    print("   ① 留出词项 ER-稠密 = %+.4f ⇒ %s" % (e - d, "ER 不差于稠密" if e >= d - 0.01 else "ER 更差"))
    ei = sum(o["interference"] for o in res["ER"]["留出词项"]) / a.seeds
    di = sum(o["interference"] for o in res["稠密"]["留出词项"]) / a.seeds
    print("   ② 吸引子干扰 ER %.4f vs 稠密 %.4f ⇒ %s" % (ei, di,
          "ER 干扰更小 (规则确实抓住了中心词)" if ei < di - 0.01 else "无机制优势"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
