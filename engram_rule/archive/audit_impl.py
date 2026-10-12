"""实现审计: 先确认"我的代码没错", 再谈架构好不好。

问三个问题 (每个都可能推翻/坐实上一轮的负面结论):
  A1 规则路径消融: 训好之后把规则路径整体置 0 ⇒ ppl 变不变? 不变说明规则路径**没被用上**
  A2 量级审计: 规则打分 vs 内容打分的幅度比 —— 若规则被淹掉就是**实现级尺度错**
  A3 规则单飞: 强制 λ=0 (关掉内容路径) 重训 ⇒ 规则路径**自己**能不能学到 CFG 的主谓一致?
     (能 ⇒ 规则路径有表达能力, 联合训练被内容路径盖住; 不能 ⇒ 规则路径先天不足)
"""
import math, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from small_corpus import cfg_corpus, CFG_TOK

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, BS, L = 300, 16, 16


def make(cfg):
    m = EngramRuleLM(cfg).to(DEV)
    m.engram.build_from_corpus(cfg_corpus(1200, L, seed=9).reshape(1, -1)[:, :20000])
    return m


def train(m, cfg, dtr, lr=3e-3, lam=None, free_scale=False):
    if lam is not None:
        for b in m.blocks:
            with torch.no_grad():
                b.attn.lam.fill_(lam)
            b.attn.lam.requires_grad_(not free_scale)
    if free_scale:
        for b in m.blocks:
            with torch.no_grad():
                b.attn.rule_scale.fill_(8.0)      # 把规则分提到与内容分同量级
    g = torch.Generator().manual_seed(4)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    m.train()
    for s in range(STEPS):
        i = torch.randint(0, dtr.shape[0], (BS,), generator=g)
        x = dtr[i].to(DEV)
        lg = m(x)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    return m


@torch.no_grad()
def ppl_of(m, cfg, dte):
    m.eval()
    x = dte.to(DEV)
    lg = m(x)
    l = F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab), x[:, 1:].reshape(-1))
    acc = (lg[:, :-1].argmax(-1) == x[:, 1:]).float().mean().item()
    m.train()
    return math.exp(min(l.item(), 20)), acc


def main():
    cfg = RCfg(vocab=len(CFG_TOK), d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 16, min_count=8, n_buckets=16)
    dtr, dte = cfg_corpus(2400, L, seed=1), cfg_corpus(400, L, seed=2)
    print("设备 %s | CFG 语料 | 步数 %d\n" % (DEV, STEPS))

    m = make(cfg); train(m, cfg, dtr)
    for b in m.blocks:
        b.attn.record = True
    p_full, a_full = ppl_of(m, cfg, dte)
    mag = dict(m.blocks[0].attn.last_mag)
    print("A2 量级审计 (训后):")
    print("   |规则分| 均值 %.6f   |内容分| 均值 %.4f   ⇒ 比值 %.2e"
          % (mag["rule_absmean"], mag["content_absmean"],
             mag["rule_absmean"] / max(mag["content_absmean"], 1e-12)))
    print("   λ=%.4f  rule_scale=%.4f  head_w 均值 %.4f  |bucket偏置| 均值 %.6f"
          % (mag["lam"], mag["rule_scale"], mag["head_w_mean"], mag["bucket_absmean"]))

    for b in m.blocks:
        b.attn.ablate_rule = True
    p_abl, a_abl = ppl_of(m, cfg, dte)
    for b in m.blocks:
        b.attn.ablate_rule = False
    print("\nA1 规则路径消融:")
    print("   完整 ppl %.4f / acc %.4f   →   规则路径置 0 后 ppl %.4f / acc %.4f   Δppl %+.5f"
          % (p_full, a_full, p_abl, a_abl, p_abl - p_full))
    print("   ⇒ %s" % ("规则路径**没被用上** (消融无差别)" if abs(p_abl - p_full) < 1e-3
                       else "规则路径确有贡献 (消融变差 %.4f)" % (p_abl - p_full)))

    print("\nA3 规则单飞 (λ=0, 只用规则路径):")
    m2 = make(cfg); train(m2, cfg, dtr, lam=0.0)
    p2, a2 = ppl_of(m2, cfg, dte)
    exp = math.log(len(CFG_TOK))
    # ★ 判据修正: 原来写死 exp*0.6 是我拍的门槛, 会把"ppl 1.63 vs 均匀 2.56"这种明显学会的情况判成"学不到"。
    #   改成与**完整模型**比: 规则单飞若接近完整模型 ⇒ 规则路径自己足以扛住任务。
    print("   ppl %.4f / acc %.4f  (均匀基线 ppl %.2f; 完整模型 ppl %.4f)"
          % (p2, a2, exp, p_full))
    print("   ⇒ %s" % ("规则路径**自己就足以**扛住这个任务 (与完整模型持平)"
                       if p2 < p_full * 1.10 else "规则路径自己不够 (劣于完整模型 %.1f%%)"
                       % ((p2 / p_full - 1) * 100)))

    print("\nA2b 尺度修正后 (rule_scale 初始放大到 8):")
    m3 = make(cfg); train(m3, cfg, dtr, free_scale=True)
    p3, a3 = ppl_of(m3, cfg, dte)
    for b in m3.blocks:
        b.attn.record = True
    p3b, _ = ppl_of(m3, cfg, dte)
    mg = m3.blocks[0].attn.last_mag
    print("   完整 ppl %.4f / acc %.4f  |规则分|/|内容分| = %.3f (原 %.2e)"
          % (p3, a3, mg["rule_absmean"] / max(mg["content_absmean"], 1e-12),
             mag["rule_absmean"] / max(mag["content_absmean"], 1e-12)))
    for b in m3.blocks:
        b.attn.ablate_rule = True
    p3c, _ = ppl_of(m3, cfg, dte)
    print("   消融后 ppl %.4f  ⇒ Δppl %+.5f" % (p3c, p3c - p3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
