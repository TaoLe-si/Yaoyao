"""小型语料测试: Engram-Rule vs 稠密基线。

语料 A (合成 CFG, 有真句法与**长距离依赖**): 主谓一致 —— 动词的单复数必须匹配主语,
  中间可插入任意层 PP 把它隔远。这正是 §4.1 那条规则("动词 → 关注左边最近名词")该起作用的地方。
  另留出 **未见词组合** 的句子做泛化测试。
语料 B (手写小型中文, 字符级): 真实文本, 规模很小, 看它在真实文本上的表现。

上一轮的三个问题这次一起处理: (a) 有句法的语料 (b) 把 min_count 调高让**空槽真实存在**
(c) 同时量 L=64 与 L=256 的墙钟 (规则路径开销不随 L 涨, 稠密注意力才涨 ⇒ 交叉点)。
"""
import argparse
import io
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from model import RCfg, EngramRuleLM
from baseline.dense import DenseLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------- 语料 A: CFG
CFG_TOK = ["the", "a", "dog", "dogs", "cat", "cats", "bird", "birds",
           "sees", "see", "chases", "chase", "."]
T = {t: i for i, t in enumerate(CFG_TOK)}
SG = {0, 2, 4, 6}          # 名词/限定词的单数索引 (the/a/dog/cat/bird 的 sg 形式)
PL = {1, 3, 5, 7}


def gen_sentence(g, max_pp=2, unseen=False):
    """NP VP, 中间插 PP; 动词单复数与主语一致; unseen=True 时用留出的词组合"""
    det = int(torch.randint(0, 2, (1,), generator=g))
    noun_sg = [2, 4, 6][int(torch.randint(0, 3, (1,), generator=g))]
    plural = bool(torch.randint(0, 2, (1,), generator=g))
    noun = noun_sg + (1 if plural else 0)
    if unseen:                       # 留出组合: 强制 cat 家族 + 特定介词序列
        noun = 4 + (1 if plural else 0)
    toks = [det, noun]
    npp = int(torch.randint(0, max_pp + 1, (1,), generator=g))
    for _ in range(npp):
        p = 8 + int(torch.randint(0, 2, (1,), generator=g))
        d2 = int(torch.randint(0, 2, (1,), generator=g))
        n2 = [2, 4, 6][int(torch.randint(0, 3, (1,), generator=g))] + \
             (1 if torch.randint(0, 2, (1,), generator=g) else 0)
        toks += [p, d2, n2]
    verb = (9 if plural else 8) if unseen else (8 + (1 if plural else 0))
    v2 = (11 if plural else 10) if unseen else (10 + (1 if plural else 0))
    toks += [verb if torch.rand(1, generator=g) < 0.5 else v2, det, noun_sg + (1 if plural else 0), 12]
    return toks


def cfg_corpus(n, L, seed=0, unseen=False):
    g = torch.Generator().manual_seed(seed)
    rows = []
    while len(rows) < n:
        s = gen_sentence(g, unseen=unseen)
        if len(s) <= L:
            rows.append(s + [12] * (L - len(s)))   # 用句号填充
    return torch.tensor(rows, dtype=torch.long)


# --------------------------------------------------------------- 语料 B: 手写中文
ZH = ("苹果是一种水果。香蕉也是一种水果。水果都能吃。蔬菜都能吃。动物不能吃。"
      "白菜是一种蔬菜。萝卜是一种蔬菜。黄瓜是一种蔬菜。猫是一种动物。狗是一种动物。"
      "老虎是一种动物。问：苹果能吃吗？答：能。问：猫能吃吗？答：不能。"
      "问：白菜能吃吗？答：能。问：苹果是什么？答：水果。问：猫是什么？答：动物。"
      "橘子是一种水果。葡萄是一种水果。问：葡萄能吃吗？答：能。问：老虎是什么？答：动物。")


def zh_corpus(L, seed=0):
    chars = sorted(set(ZH))
    itos = ["<pad>"] + chars
    stoi = {s: i for i, s in enumerate(itos)}
    ids = [stoi[c] for c in ZH]
    rows = [ids[i:i + L] for i in range(0, len(ids) - L, max(1, L // 2))]
    return torch.tensor(rows, dtype=torch.long), len(itos), itos


# --------------------------------------------------------------- 训练/评测
def train_eval(model, dtr, dte, vocab, L, steps, bs, lr, tag):
    g = torch.Generator().manual_seed(3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    t0 = time.time()
    for s in range(1, steps + 1):
        i = torch.randint(0, dtr.shape[0], (bs,), generator=g)
        x = dtr[i].to(DEV)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if s % max(1, steps // 4) == 0 or s == 1:
            print("    [%s] %4d/%d loss %.4f ppl %.1f  %.1fs"
                  % (tag, s, steps, loss.item(), math.exp(min(loss.item(), 20)), time.time() - t0))
            sys.stdout.flush()
    model.eval()
    with torch.no_grad():
        i = torch.randint(0, dte.shape[0], (min(256, dte.shape[0]),), generator=g)
        x = dte[i].to(DEV)
        lg = model(x)
        l = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        acc = (lg[:, :-1].argmax(-1) == x[:, 1:]).float().mean().item()
    # 速度 (两个长度; 规则路径不随 L 涨, 稠密注意力才涨)
    sp = {}
    for Lq in (L, L * 4):
        xq = torch.randint(0, vocab, (1, Lq)).to(DEV)
        with torch.no_grad():
            for _ in range(2): model(xq)
            if DEV == "cuda": torch.cuda.synchronize()
            t1 = time.time()
            for _ in range(5): model(xq)
            if DEV == "cuda": torch.cuda.synchronize()
            sp[Lq] = 5 * Lq / (time.time() - t1)
    return dict(ppl=math.exp(min(l.item(), 20)), acc=acc, sp=sp)


def run_corpus(name, dtr, dte, vocab, args, extra_te=None):
    cfg = RCfg(vocab=vocab, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               ngram_max=3, n_hash=3, n_slots=1 << 16, min_count=args.min_count, n_buckets=16)
    print("\n================ 语料 %s ================" % name)
    print("  训练 %d 行 / 测试 %d 行, 序列长 %d, 词表 %d" % (dtr.shape[0], dte.shape[0], dtr.shape[1], vocab))
    er = EngramRuleLM(cfg).to(DEV)
    info = er.engram.build_from_corpus(dtr.reshape(1, -1)[:, :100_000])
    print("  Engram 初始化(min_count=%d): 存活槽 %d/%d = %.2f%%"
          % (args.min_count, info["alive_slots"], info["n_slots"], info["alive_ratio"] * 100))
    dn = DenseLM(cfg).to(DEV)
    print("  [Engram-Rule]")
    r_er = train_eval(er, dtr, dte, vocab, dtr.shape[1], args.steps, args.bs, args.lr, "ER")
    print("  [稠密基线]")
    r_dn = train_eval(dn, dtr, dte, vocab, dtr.shape[1], args.steps, args.bs, args.lr, "DN")
    with torch.no_grad():
        er(dte[:32].to(DEV))
    st = er.stats()
    gen = ""
    if extra_te is not None:
        with torch.no_grad():
            x = extra_te.to(DEV)
            lg = er(x)
            l = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
            acc = (lg[:, :-1].argmax(-1) == x[:, 1:]).float().mean().item()
        with torch.no_grad():
            lgd = dn(x)
            ld = F.cross_entropy(lgd[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
            accd = (lgd[:, :-1].argmax(-1) == x[:, 1:]).float().mean().item()
        gen = ("  ★泛化(未见词组合) ER ppl %.2f/acc %.4f | 稠密 ppl %.2f/acc %.4f"
               % (math.exp(min(l.item(), 20)), acc, math.exp(min(ld.item(), 20)), accd))
    print("  ---- 指标 ----")
    print("  困惑度      ER %.3f | 稠密 %.3f" % (r_er["ppl"], r_dn["ppl"]))
    print("  准确率      ER %.4f | 稠密 %.4f" % (r_er["acc"], r_dn["acc"]))
    for Lq in r_er["sp"]:
        print("  tokens/s L=%-4d ER %6.0f | 稠密 %6.0f | 比值 %.2f×"
              % (Lq, r_er["sp"][Lq], r_dn["sp"][Lq], r_er["sp"][Lq] / r_dn["sp"][Lq]))
    print("  稀疏度      存活槽 %.2f%%, 规则路径跳过率 %.3f" %
          (100 * st["alive_slots"] / st["n_slots"], st["rule_skip_ratio"]))
    print("  参数        ER %.3fM | 稠密 %.3fM" % (sum(p.numel() for p in er.parameters()) / 1e6,
                                                 sum(p.numel() for p in dn.parameters()) / 1e6))
    if gen:
        print(gen)
    return r_er, r_dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min_count", type=int, default=8)
    a = ap.parse_args()
    print("设备 %s | 优化器 AdamW | 步数 %d" % (DEV, a.steps))
    L = 16
    dtr = cfg_corpus(2400, L, seed=1)
    dte = cfg_corpus(400, L, seed=2)
    un = cfg_corpus(200, L, seed=3, unseen=True)
    run_corpus("A 合成 CFG (主谓一致 + 长距离依赖)", dtr, dte, len(CFG_TOK), a, extra_te=un)
    z, vz, _ = zh_corpus(L)
    run_corpus("B 手写中文 (字符级)", z[:len(z) * 3 // 4], z[len(z) * 3 // 4:], vz, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
