"""扎住的评测 (替代 small_corpus.py 里那套有缺陷的设置)。

静态分析查出的四点, 逐条修:
  E1 语料 B 只有 15 训练/6 测试行 ⇒ 读数被噪声主导。改为**程序生成的大号模板中文语料**
     (900/300 行), 并断言训练/测试**无泄漏**。
  E2 单种子 ⇒ 无法区分"模型差异"与"种子方差"。改为 **3 种子, 报 均值±标准差**。
  E3 ER 参数 17.4M vs 稠密 0.53M, 固定步数下不公平 ⇒ 补一个**参数匹配**的稠密基线
     (自动搜 d 使参数量对齐)。
  E4 速度扫描用 alive[:]=True (最大工作量) ⇒ 改为按**语料初始化的真实存活率**测。

跑法: python engram_rule/eval_solid.py --steps 200 --seeds 3
"""
import argparse
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

CFG_TOK = ["the", "a",
           "dog", "dogs", "cat", "cats", "bird", "birds", "horse", "horses", "fish", "fishes",
           "in", "on", "near", "with",
           "sees", "see", "chases", "chase", "likes", "like",
           "."]
DET = [0, 1]
N_SG = [2, 4, 6, 8, 10]
N_PL = [3, 5, 7, 9, 11]
PREP = [12, 13, 14, 15]
V_SG = [16, 18, 20]
V_PL = [17, 19, 21]
END = 22
# ★ 原版只有 3 个名词 + 2 个介词, 熵不够 ⇒ 2400 训练行与 400 测试行**撞了 50 行**(真泄漏, 检查 H2 抓出)。
#   现在名词 5 对、介词 4 个、动词 3 对, 并加**拒绝采样**保证三集合逐行不相交。


def gen_sentence(g, max_pp=2, unseen=False):
    det = DET[int(torch.randint(0, len(DET), (1,), generator=g))]
    j = int(torch.randint(0, len(N_SG), (1,), generator=g))
    plural = bool(torch.randint(0, 2, (1,), generator=g))
    noun = (N_PL if plural else N_SG)[0 if unseen else j]
    toks = [det, noun]
    for _ in range(int(torch.randint(0, max_pp + 1, (1,), generator=g))):
        p = PREP[int(torch.randint(0, len(PREP), (1,), generator=g))]
        d2 = DET[int(torch.randint(0, len(DET), (1,), generator=g))]
        j2 = int(torch.randint(0, len(N_SG), (1,), generator=g))
        pl2 = bool(torch.randint(0, 2, (1,), generator=g))
        toks += [p, d2, (N_PL if pl2 else N_SG)[j2]]
    vs = V_PL if plural else V_SG
    v = vs[(1 if unseen else 0) % len(vs)]
    toks += [v, det, (N_PL if plural else N_SG)[j], END]
    return toks


def cfg_corpus(n, L, seed=0, unseen=False, avoid=None):
    """★ 拒绝采样: 与 avoid 里已有的行**逐行不相交** (旧版允许重复 ⇒ 训练/测试撞 50 行)"""
    g = torch.Generator().manual_seed(seed)
    rows, seen = [], set(avoid or ())
    tries = 0
    while len(rows) < n and tries < n * 400:
        tries += 1
        s = gen_sentence(g, unseen=unseen)
        if len(s) > L:
            continue
        row = tuple(s + [END] * (L - len(s)))
        if row in seen:
            continue
        seen.add(row)
        rows.append(list(row))
    assert len(rows) == n, "语料熵不足: 只生成了 %d/%d 行" % (len(rows), n)
    return torch.tensor(rows, dtype=torch.long)


# --------------------------------------------------------------- 语料 B: 大号模板中文
ZH_SUBJ = ["苹果", "香蕉", "橘子", "葡萄", "桃子", "梨", "西瓜", "草莓"]
ZH_CLS = ["水果", "蔬菜", "动物", "工具", "乐器"]
ZH_OTHER = ["白菜", "萝卜", "黄瓜", "番茄", "菠菜", "猫", "狗", "兔子", "大象",
            "锤子", "扳手", "螺丝刀", "钢琴", "小提琴", "笛子"]
ZH_PROP = ["能吃", "不能吃", "是活的", "不是活的", "很甜", "很酸"]


def zh_sentence(g):
    """模板 + 随机组合, 规模足够做统计 (但仍是"小型语料"量级)"""
    s = ZH_SUBJ[int(torch.randint(0, len(ZH_SUBJ), (1,), generator=g))]
    c = ZH_CLS[int(torch.randint(0, len(ZH_CLS), (1,), generator=g))]
    o = ZH_OTHER[int(torch.randint(0, len(ZH_OTHER), (1,), generator=g))]
    p = ZH_PROP[int(torch.randint(0, len(ZH_PROP), (1,), generator=g))]
    forms = ["%s是一种%s。" % (s, c),
             "%s也是一种%s。" % (o, c),
             "问：%s是什么？答：%s。" % (s, c),
             "问：%s%s吗？答：%s。" % (s, p, "能" if p in ("能吃", "很甜") else "不能"),
             "问：%s能吃吗？答：能。" % s,
             "%s%s。" % (c, p)]
    return forms[int(torch.randint(0, len(forms), (1,), generator=g))]


ZH_FIXED = "".join(ZH_SUBJ + ZH_CLS + ZH_OTHER + ZH_PROP) +     "是一种也是一种。问：什么？答：吗？能不能"
ZH_CHARS = ["<pad>"] + sorted(set(ZH_FIXED))
ZH_STOI = {c: i for i, c in enumerate(ZH_CHARS)}


def zh_corpus(n_rows, L, seed=0, avoid=None):
    """★ 路线4（第二版）: 旧版是**连续文本切片** ⇒ 训练/测试 21 行重叠。
    第一版修法要求"句子级去重且互不相同"，但模板组合只有约 241 种，
    却要 n_rows×L 个不同句子（1400×16=22400）⇒ **while 永真、死循环**（静态可算出的错）。
    本版: 句子可重复, 只对**行**去重（泄漏的关键就是行不能重）, 且循环带**硬上限+断言**。
    另: 用**固定字符表**（旧版每次调用重建词表 ⇒ 训练/测试 id 含义不一致）。"""
    g = torch.Generator().manual_seed(seed)
    seen, rows = set(avoid or ()), []
    budget = n_rows * 40 + 1000                    # 硬上限: 宁可报错也不许空转
    while len(rows) < n_rows and budget > 0:
        text = "".join(zh_sentence(g) for _ in range(8))
        ids = [ZH_STOI[c] for c in text]
        for i in range(len(ids) - L + 1):
            r = tuple(ids[i:i + L])
            if r in seen:
                continue
            seen.add(r); rows.append(list(r)); budget -= 1
            if len(rows) >= n_rows:
                break
    assert len(rows) == n_rows, "中文语料只生成 %d/%d 行 (预算耗尽)" % (len(rows), n_rows)
    return torch.tensor(rows, dtype=torch.long), len(ZH_CHARS)


# --------------------------------------------------------------- 训练/评测
def train_eval(model, dtr, dte, vocab, L, steps, bs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train()
    for _ in range(steps):
        i = torch.randint(0, dtr.shape[0], (bs,), generator=g)
        x = dtr[i].to(DEV)
        lg = model(x)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    model.eval()
    with torch.no_grad():
        x = dte.to(DEV)
        lg = model(x)
        l = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
        acc = (lg[:, :-1].argmax(-1) == x[:, 1:]).float().mean().item()
    return math.exp(min(l.item(), 20)), acc


def matched_dense_cfg(vocab, target_params, layers=4, heads=4, ffn_mult=2):
    """搜一个 d, 让稠密模型的参数量与 ER **对齐** (E3)"""
    best, bd = None, None
    for d in range(64, 2049, 64):
        if d % heads:
            continue
        c = RCfg(vocab=vocab, d=d, n_layers=layers, n_heads=heads, ffn_mult=ffn_mult,
                 rope_dim=max(2, d // heads // 2))
        with torch.device("meta"):          # 只数参数, 不真分配 (否则 31 个候选会吃掉几百 MB)
            n = sum(p.numel() for p in DenseLM(c).parameters())
        if bd is None or abs(n - target_params) < bd:
            best, bd = c, abs(n - target_params)
    return best


def speed(model, L, live_ratio):
    # ★ 不能用写死的词表大小: CFG 语料只有 13 个 token, 写死 256 会越界触发 device-side assert
    V = model.emb.num_embeddings if hasattr(model, "emb") else 256
    x = torch.randint(0, V, (1, L)).to(DEV)
    with torch.no_grad():
        model(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            model(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
    return 3 * L / (time.time() - t), live_ratio


def run(name, dtr, dte, vocab, args, unseen=None, L=16):
    print("\n================ %s ================" % name)
    assert dte.shape[0] >= 30 or unseen is not None, "E1: 测试行太少"
    # E1: 无泄漏
    tr = set(map(tuple, dtr.tolist()))          # ★ 旧版只查前 400 行 ⇒ 漏掉了 50 行真泄漏
    te = set(map(tuple, dte.tolist()))
    leak = len(tr & te)
    print("  E1 训练 %d 行 / 测试 %d 行, 逐行重叠(前 400 训练行 vs 测试) = %d" % (dtr.shape[0], dte.shape[0], leak))
    cfg = RCfg(vocab=vocab, d=128, n_layers=4, n_heads=4, ffn_mult=2, rope_dim=16,
               n_slots=1 << 16, min_count=args.min_count, n_buckets=16)
    res = {}
    live = 0.0
    er_all, dn_all = [], []
    for seed in range(args.seeds):          # E2: 多种子
        er = EngramRuleLM(cfg).to(DEV)
        er.engram.build_from_corpus(dtr.reshape(1, -1)[:, :100_000])
        dn = DenseLM(cfg).to(DEV)
        p_er, a_er = train_eval(er, dtr, dte, vocab, L, args.steps, args.bs, args.lr, seed)
        p_dn, a_dn = train_eval(dn, dtr, dte, vocab, L, args.steps, args.bs, args.lr, seed)
        res.setdefault("er", []).append((p_er, a_er))
        res.setdefault("dn", []).append((p_dn, a_dn))
        live = float(er.engram.alive.float().mean())
        er_all.append(er); dn_all.append(dn)
        if seed == 0:
            er0, dn0 = er, dn
    # E3: 参数匹配的稠密基线
    n_er = sum(p.numel() for p in er0.parameters())
    mcfg = matched_dense_cfg(vocab, n_er, layers=4, heads=4, ffn_mult=2)
    dm = DenseLM(mcfg).to(DEV)
    n_dm = sum(p.numel() for p in dm.parameters())
    p_dm, a_dm = train_eval(dm, dtr, dte, vocab, L, args.steps, args.bs, args.lr, 0)

    def ms(v):
        ps = [x[0] for x in v]; asc = [x[1] for x in v]
        return (sum(ps) / len(ps), max(ps) - min(ps), sum(asc) / len(asc))

    e_p, e_s, e_a = ms(res["er"]); d_p, d_s, d_a = ms(res["dn"])
    print("  E2 %d 种子:" % args.seeds)
    print("     Engram-Rule  ppl %.4f (极差 %.4f)  acc %.4f" % (e_p, e_s, e_a))
    print("     稠密(同结构)  ppl %.4f (极差 %.4f)  acc %.4f" % (d_p, d_s, d_a))
    print("     稠密(匹配参数 d=%d, %.3fM) ppl %.4f acc %.4f  [ER %.3fM]"
          % (mcfg.d, n_dm / 1e6, p_dm, a_dm, n_er / 1e6))
    print("     ⇒ ppl 差 %+.4f (ER 好为正), 种子极差 %.4f ⇒ %s"
          % (d_p - e_p, max(e_s, d_s),
             "差异**小于**种子方差, 不能作结论" if abs(d_p - e_p) < max(e_s, d_s) else "差异大于种子方差"))
    # E4: 真实存活率下测速
    print("  E4 真实存活率 %.2f%% 下测速:" % (live * 100))
    for Lq in (16, 64, 256):
        s_er, _ = speed(er0, Lq, live); s_dn, _ = speed(dn0, Lq, live)
        print("     L=%-4d ER %7.0f | 稠密(同结构) %7.0f | 比值 %.2f×" % (Lq, s_er, s_dn, s_er / s_dn))
    if unseen is not None:
        # ★ 路线5: 旧版只用了 seed 0 的模型 ⇒ 改成**全部种子的均值**
        acc = {}
        with torch.no_grad():
            x = unseen.to(DEV)
            for tag, ms in (("er", er_all), ("dn", dn_all)):
                ps, as_ = [], []
                for mm in ms:
                    lg = mm(x)
                    ps.append(math.exp(min(F.cross_entropy(
                        lg[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1)).item(), 20)))
                    as_.append((lg[:, :-1].argmax(-1) == x[:, 1:]).float().mean().item())
                acc[tag] = (sum(ps) / len(ps), max(ps) - min(ps), sum(as_) / len(as_))
        print("  泛化(未见词组合 %d 行, %d 种子均值): ER ppl %.3f(极差 %.3f)/acc %.4f | 稠密 ppl %.3f(极差 %.3f)/acc %.4f"
              % (unseen.shape[0], args.seeds, acc["er"][0], acc["er"][1], acc["er"][2],
                 acc["dn"][0], acc["dn"][1], acc["dn"][2]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=256)   # ★ 性能: bs=16 时 GPU 只 46% 利用率, 256→71%, 1024→94%
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min_count", type=int, default=8)
    a = ap.parse_args()
    print("设备 %s | 步数 %d | 种子 %d" % (DEV, a.steps, a.seeds))
    L = 16
    dtr = cfg_corpus(2400, L, seed=1)
    dte = cfg_corpus(400, L, seed=2, avoid=set(map(tuple, dtr.tolist())))
    dun = cfg_corpus(200, L, seed=3, unseen=True,
                     avoid=set(map(tuple, dtr.tolist())) | set(map(tuple, dte.tolist())))
    run("A 合成 CFG (主谓一致 + 长距离依赖)", dtr, dte, len(CFG_TOK), a, unseen=dun, L=L)
    dtr, dte, vz, _ = None, None, None, None
    ztr, vz = zh_corpus(1400, L, seed=11)
    zte, _ = zh_corpus(450, L, seed=12, avoid=set(map(tuple, ztr.tolist())))
    run("B 模板中文 (句子级不相交 %d/%d)" % (ztr.shape[0], zte.shape[0]), ztr, zte, vz, a, L=L)
    return 0


if __name__ == "__main__":
    sys.exit(main())
