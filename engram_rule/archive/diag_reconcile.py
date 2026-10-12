"""查清 §8.3 与 §8.5 的干扰数字矛盾: 同一次运行、同一批行, 同时量两样。
  A 整模型输出(全部 500 行): acc / 无吸引子 / 吸引子 / 干扰
  B 第 0 层门的归因(前 128 行): 中心词份额 / 吸引子份额
若 A 与 §8.3 一致、B 与 §8.5 一致 ⇒ 矛盾只是**口径**(行数/层数)不同;
若同一批行上两样互相矛盾 ⇒ 某一边的算法有误。"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from corpus_agreement import corpus, T, VOCAB, V_SG, V_PL

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def head_off(x, vp):
    out = []
    for i in range(x.shape[0]):
        v = int(vp[i])
        out.append(1 + 3 * (sum(1 for j in range(v) if int(x[i, j]) == T["the"]) - 1))
    return torch.tensor(out)


def main():
    L = 24
    cfg = RCfg(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=16, gate_rank=16)
    xtr, vtr, _, _ = corpus(2400, L, seed=1)
    xh, vh, ah, hh = corpus(500, L, seed=3, hold_nouns=True, hold_verbs=True,
                            avoid=set(map(tuple, xtr.tolist())))
    dh = head_off(xh, vh)
    V1 = torch.tensor([T[v] for v in V_SG], device=DEV)
    V2 = torch.tensor([T[v] for v in V_PL], device=DEV)
    print("设备 %s | 同一批行: A=全模型输出(全部 %d 行), B=第0层门归因(前 128 行)" % (DEV, xh.shape[0]))
    print("  %-5s %7s %7s %7s %9s | %9s %9s %s"
          % ("种子", "acc", "无吸引", "吸引子", "干扰(A)", "中心词份额", "吸引子份额", "门(B)"))
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
        m.eval()
        a = m.blocks[0].attn
        a.record = True                                    # 同一批行, 一次前向两样都读
        with torch.no_grad():
            xd = xh.to(DEV)
            lg = m(xd)                                     # (500, L, vocab)
        p = lg[torch.arange(500), vh].softmax(-1)
        corr = torch.where(hh.to(DEV), p[:, V1].sum(-1) > p[:, V2].sum(-1),
                           p[:, V2].sum(-1) > p[:, V1].sum(-1))
        at = ah.to(DEV)
        acc = float(corr.float().mean())
        na = float(corr[~at].float().mean())
        aa = float(corr[at].float().mean())
        contrib = a.last_contrib[:128]                      # (128, L, W) → 取动词行
        rows = (torch.arange(128), vh[:128])
        w = contrib[rows].abs() + 1e-9
        w = w / w.sum(-1, keepdim=True)
        atm = ah[:128]
        hs = float(w[torch.arange(128), dh[:128]][atm].mean())
        as_ = float(w[torch.arange(128), torch.ones(128, dtype=torch.long)][atm].mean())
        print("  %-5d %7.4f %7.4f %7.4f %9.4f | %9.4f %9.4f %s"
              % (sd, acc, na, aa, na - aa, hs, as_, "中心词" if hs > as_ else "**吸引子**"))
        sys.stdout.flush()
        m.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
