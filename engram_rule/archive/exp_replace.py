"""SPEC v1.1 §7-1 实验: 内容路径真替换能不能兑现?
  静态分析结论: 关门在损失上**没有动力**(规则路径单独就够), 必须加内容使用率惩罚 ρ·mean(gc)。
  本实验对照 ρ=0 / ρ>0: 门均值、质量(ppl/acc)、以及**关门后墙钟是否真的变快**。"""
import argparse, math, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RCfg, EngramRuleLM
from eval_solid import cfg_corpus, CFG_TOK

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train(cfg, dtr, rho, steps, L, seed=0, bs=16, lr=3e-3):
    torch.manual_seed(seed)            # ★ 两个对照必须同种子 (C13 治过的错)
    m = EngramRuleLM(cfg).to(DEV)
    m.engram.build_from_corpus(dtr.reshape(1, -1)[:, :50000])
    for b in m.blocks:
        b.attn.record_gc = True
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    m.train()
    hist = []
    for s in range(steps):
        i = torch.randint(0, dtr.shape[0], (bs,), generator=g)
        x = dtr[i].to(DEV)
        lg = m(x)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab), x[:, 1:].reshape(-1))
        pen = rho * m.content_usage(grad=True)      # ★ 必须用可微版本 (C12)
        (loss + pen).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); sch.step()
        if s % max(1, steps // 4) == 0 or s == steps - 1:
            hist.append((s, loss.item(), m.content_usage()))
    return m, hist


@torch.no_grad()
def measure(m, cfg, dte, L):
    m.eval()
    x = dte.to(DEV)
    lg = m(x)
    l = F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab), x[:, 1:].reshape(-1))
    acc = (lg[:, :-1].argmax(-1) == x[:, 1:]).float().mean().item()
    gc = m.content_usage()
    # 墙钟: 关掉跳过 vs 打开跳过
    def speed(use_skip):
        m.set_sparse_content(use_skip, 0.05)
        for Lq in (64, 256):
            xq = torch.randint(0, cfg.vocab, (1, Lq)).to(DEV)
            m(xq)
            if DEV == "cuda": torch.cuda.synchronize()
            t = time.time()
            for _ in range(3): m(xq)
            if DEV == "cuda": torch.cuda.synchronize()
            yield Lq, 3 * Lq / (time.time() - t)
    a = dict(speed(False)); b = dict(speed(True))
    m.set_sparse_content(False)
    m.train()
    return math.exp(min(l.item(), 20)), acc, gc, a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rho", type=float, default=0.5)
    a = ap.parse_args()
    L = 16
    cfg = RCfg(vocab=len(CFG_TOK), d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 16, min_count=8, n_buckets=16)
    dtr = cfg_corpus(2400, L, seed=1)
    dte = cfg_corpus(400, L, seed=2, avoid=set(map(tuple, dtr.tolist())))
    print("设备 %s | CFG %d/%d | 步数 %d | 门阈值 τ=0.05" % (DEV, dtr.shape[0], dte.shape[0], a.steps))
    for rho in (0.0, a.rho):
        m, hist = train(cfg, dtr, rho, a.steps, L)
        ppl, acc, gc, sp0, sp1 = measure(m, cfg, dte, L)
        print("\n  ρ=%.2f  门均值 %.4f (关闭位置占比 %.1f%%)  ppl %.4f  acc %.4f"
              % (rho, gc, 100 * (gc < 0.05) if gc < 0.05 else max(0.0, 100 * (1 - gc)), ppl, acc))
        print("    训练轨迹 (步, loss, 门均值): %s" % [(s, round(l, 3), round(g, 3)) for s, l, g in hist])
        for Lq in (64, 256):
            print("    L=%-4d 不跳过 %7.0f tok/s | 跳过 %7.0f tok/s | 提速 %.3f×"
                  % (Lq, sp0[Lq], sp1[Lq], sp1[Lq] / sp0[Lq]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
