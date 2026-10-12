"""★ 验证 100% 是不是真的: 反事实换头名词。
若模型真在读头名词的"数": 换成对调/常数/随机 token 后准确率必须掉。
若不掉 ⇒ 100% 是泄漏, "平局"判决作废。判据可失败(有明确的掉分预期)。"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from baseline.dense import DenseLM
from corpus_agreement import corpus, T, VOCAB, V_SG, V_PL, N_SG, N_PL
from exp_agreement import run as train_run

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def acc(m, x, vp, hs):
    m.eval()
    V1 = torch.tensor([T[v] for v in V_SG], device=DEV)
    V2 = torch.tensor([T[v] for v in V_PL], device=DEV)
    with torch.no_grad():
        lg = m(x.to(DEV))
    p = lg[torch.arange(x.shape[0]), vp].softmax(-1)
    corr = torch.where(hs.to(DEV), p[:, V1].sum(-1) > p[:, V2].sum(-1),
                       p[:, V2].sum(-1) > p[:, V1].sum(-1))
    m.train()
    return float(corr.float().mean())


def swap_head(x, hs, sg_tok, pl_tok):
    """把每行位置 1 的头名词按其单复数换成给定 token (sg_tok 对应单数, pl_tok 对应复数)"""
    x2 = x.clone()
    for i in range(x.shape[0]):
        x2[i, 1] = T[sg_tok] if bool(hs[i]) else T[pl_tok]
    return x2


def main():
    L = 24
    cfg = RCfg(vocab=VOCAB, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 14, min_count=4, n_buckets=64, k_buckets=4, window_max=16, gate_rank=16)
    xtr, vtr, _, htr = corpus(2400, L, seed=1)
    xh, vh, ah, hh = corpus(500, L, seed=3, hold_nouns=True, hold_verbs=False,
                            avoid=set(map(tuple, xtr.tolist())))
    for tag, mk in (("ER", lambda: EngramRuleLM(cfg).to(DEV)),
                    ("稠密", lambda: DenseLM(cfg).to(DEV))):
        torch.manual_seed(0)
        m = mk()
        if tag == "ER":
            m.engram.build_from_corpus(xtr.reshape(1, -1)[:, :50000])
        train_run(m, xtr, vtr, htr, 400, 256, 3e-3, 0)
        print("\n===== %s: 反事实换头名词 (判据: 若真在读名词, 掉分必须显著) =====" % tag)
        a0 = acc(m, xh, vh, hh)
        print("  原样 (头=未见 goat/goats|pig/pigs):        %.4f" % a0)
        # 1) 对调: 单数头 ↔ 复数头 (正确答案也应对调 ⇒ 准确率应≈0)
        x1 = swap_head(xh, ~hh, "goats", "goat")       # 故意放错边
        a1 = acc(m, x1, vh, ~hh)
        print("  头名词放错边 (goat↔goats 对调):            %.4f  (读名词 ⇒ 应≈0)" % a1)
        # 2) 换成见过的名词对 (dog/dogs, 训练里见过; 答案不变)
        x2 = swap_head(xh, hh, "dog", "dogs")
        a2 = acc(m, x2, vh, hh)
        print("  头换成**见过**的 dog/dogs:                 %.4f  (若名词被读, 应保持高)" % a2)
        # 3) 头换成常数 token (与数无关, e.g. 'the')
        x3 = xh.clone()
        x3[:, 1] = T["the"]
        a3 = acc(m, x3, vh, hh)
        print("  头抹成常数 'the':                          %.4f  (若名词被读, 应掉到≈0.5)" % a3)
        # 4) 头换成未见**动词**(完全无关的随机嵌入)
        x4 = swap_head(xh, hh, "likes", "like")
        a4 = acc(m, x4, vh, hh)
        print("  头换成未见动词 likes/like:                 %.4f" % a4)
        ok = (a1 < 0.1) or (a3 < 0.7) or (a4 < 0.7)
        print("  ⇒ %s" % ("模型确实在读头名词 (有反事实效应)" if ok else
                          "**★准确率不随头名词变 ⇒ 100% 是泄漏, 不是读了名词**"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
