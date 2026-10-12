"""★ 决定性确认: 模型是在"抄自己位置上的动词"还是在"读上下文"?
把动词 token 本身换成**错数**的动词:
  抄写模型 ⇒ 输出那个错数动词 ⇒ 准确率 ≈ 0
  读上下文 ⇒ 输出正确数动词   ⇒ 准确率 ≈ 1
另: 查询点改到动词**前一位** (vp-1, 动词不在自己的注意力窗内) ⇒ 这才是干净测法。"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from baseline.dense import DenseLM
from corpus_agreement import corpus, T, VOCAB, V_SG, V_PL
from exp_agreement import run as train_run

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def number_acc_at(m, x, vp, hs, qpos):
    """在 qpos 取 logits, 判单复数。qpos=vp 是旧口径(有抄写泄漏); qpos=vp-1 是干净口径。"""
    m.eval()
    V1 = torch.tensor([T[v] for v in V_SG], device=DEV)
    V2 = torch.tensor([T[v] for v in V_PL], device=DEV)
    with torch.no_grad():
        lg = m(x.to(DEV))
    q = torch.clamp(vp - (0 if qpos == "vp" else 1), min=0)
    p = lg[torch.arange(x.shape[0]), q].softmax(-1)
    corr = torch.where(hs.to(DEV), p[:, V1].sum(-1) > p[:, V2].sum(-1),
                       p[:, V2].sum(-1) > p[:, V1].sum(-1))
    m.train()
    return float(corr.float().mean())


def main():
    L = 24
    cfg = RCfg(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=16, gate_rank=16)
    xtr, vtr, _, htr = corpus(2400, L, seed=1)
    xh, vh, ah, hh = corpus(500, L, seed=3, hold_nouns=True, hold_verbs=False,
                            avoid=set(map(tuple, xtr.tolist())))
    Vsg = {T[v] for v in V_SG}
    for tag, mk in (("ER", lambda: EngramRuleLM(cfg).to(DEV)),
                    ("稠密", lambda: DenseLM(cfg).to(DEV))):
        torch.manual_seed(0)
        m = mk()
        if tag == "ER":
            m.engram.build_from_corpus(xtr.reshape(1, -1)[:, :50000])
        train_run(m, xtr, vtr, htr, 400, 256, 3e-3, 0)
        print("\n===== %s =====" % tag)
        print("  旧口径 (在动词位 vp 查询):                %.4f" % number_acc_at(m, xh, vh, hh, "vp"))
        print("  动词换成**错数**动词后再查 (在 vp):        ", end="")
        xw = xh.clone()
        for i in range(xh.shape[0]):
            tok = int(xh[i, int(vh[i])])
            pool = [T[v] for v in V_PL] if tok in Vsg else [T[v] for v in V_SG]
            xw[i, int(vh[i])] = pool[0]                       # 换成错数的动词
        print("%.4f  (抄写⇒0, 读上下文⇒1)" % number_acc_at(m, xw, vh, hh, "vp"))
        print("  干净口径 (**在 vp-1 查询**, 动词不在窗内):   %.4f"
              % number_acc_at(m, xh, vh, hh, "vp-1"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
