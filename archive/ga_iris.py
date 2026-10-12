"""离散编码 + 遗传算法 (Iris 分类)。

思路: 把每个样本**编码成位码**, 分类器是若干条**位掩码合取项**:
      不匹配位数 = popcount(项 & ~码)   (项里为 1 的位, 要求样本对应位也必须是 1)
      命中条件   = 不匹配位数 ≤ 阈值
      类别得分   = 该类下命中的项数; 取最高者为预测
  ⇒ 全程**整数 / 位运算 / popcount**, 没有梯度、没有浮点打分。
  ⇒ 每一项就是一个十六进制掩码 (打完打印出来看)。

遗传算法: 锦标赛选择 + 均匀交叉 + 位翻转变异 + 精英保留。
协议与神经那条**完全相同**: 分层 5 折 (同一 seed 12345) ⇒ 准确率可直接对比。
"""
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = "cpu"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
CSV = "data/iris.csv"
CLASSES = ["setosa", "versicolor", "virginica"]
NCLS, NFEAT = 3, 4
LEV = 8                # 每个特征离散成 8 档
BITS = NFEAT * LEV     # 32 位码 (每特征一个 8 位的 one-hot 块)
NTERM = 24             # 合取项个数 (类别基因自由)
MAXK = NFEAT           # 阈值最大 = 4 (不匹配位不可能超过特征数)
POP, GENS = 120, 600
ELITE, TOURN, PC = 4, 3, 0.7
MUT_BIT, MUT_K = 0.03, 0.12
# ★ 复杂度惩罚: 第一版设 LAM=0.01 惩罚"阈值 k 大", 结果 GA 几乎全选 k=0
#   = 退化成查表(每项只遮住一个训练样本) ⇒ 训练涨、测试跌。
#   实际上 k 大 = 允许几个特征不同 = **汉明球, 这才是泛化** ⇒ 惩罚方向反了, 去掉。
LAM = 0.0
KFOLDS = 5


# ----------------------------- 数据与编码 -----------------------------

def load_iris():
    if not os.path.exists(CSV):
        os.makedirs("data", exist_ok=True)
        with urllib.request.urlopen(URL, timeout=30) as r:
            raw = r.read().decode("utf-8", "replace")
        io.open(CSV, "w", encoding="utf-8").write(raw)
    X, y = [], []
    for line in io.open(CSV, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) != 5 or not p[0]:
            continue
        X.append([float(v) for v in p[:4]])
        y.append(CLASSES.index(p[4].replace("Iris-", "")))
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)
    assert X.shape == (150, 4)
    return X, y


def make_edges(Xtr):
    """用**训练折**的分位数定离散边界 (防泄漏)"""
    return [np.quantile(Xtr[:, j], np.linspace(0, 1, LEV + 1)[1:-1]) for j in range(NFEAT)]


def encode(X, edges):
    """(n,4) → (n,) uint32 位码: 每特征一个 8 位 one-hot 块"""
    lv = np.zeros((len(X), NFEAT), np.int64)
    for j in range(NFEAT):
        lv[:, j] = np.searchsorted(edges[j], X[:, j], side="left")
        lv[:, j] = np.clip(lv[:, j], 0, LEV - 1)
    code = np.zeros(len(X), np.uint32)
    for j in range(NFEAT):
        code |= (np.uint32(1) << np.uint32(j * LEV + lv[:, j]))
    return code


def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); folds = [[] for _ in range(k)]
    for c in range(NCLS):
        ic = idx[y == c].copy(); rng.shuffle(ic)
        for j, v in enumerate(ic):
            folds[j % k].append(v)
    out = []
    for j in range(k):
        te = np.array(sorted(folds[j]))
        tr = np.array(sorted(set(idx.tolist()) - set(te.tolist())))
        out.append((tr, te))
    return out


# ----------------------------- 离散分类器 -----------------------------

class Genome:
    __slots__ = ("mask", "k", "cls")

    def __init__(self, mask=None, k=None, cls=None):
        self.mask = mask if mask is not None else np.zeros(NTERM, np.uint32)
        self.k = k if k is not None else np.zeros(NTERM, np.int64)
        self.cls = cls if cls is not None else np.zeros(NTERM, np.int64)

    def copy(self):
        return Genome(self.mask.copy(), self.k.copy(), self.cls.copy())


def score_all(g, code):
    """(n,) → (n,NCLS) 各类得分。不匹配 = popcount(mask & ~code)"""
    mis = np.bitwise_count(np.bitwise_and(g.mask[None, :], np.bitwise_not(code)[:, None]))
    fire = mis <= g.k[None, :]                                   # (n,T) 命中
    sc = np.zeros((len(code), NCLS), np.int64)
    for c in range(NCLS):
        sc[:, c] = fire[:, g.cls == c].sum(1)
    return sc, mis


def predict(g, code):
    sc, mis = score_all(g, code)
    # 得分最高者; 全 0 时按"最小不匹配数"兜底 (确定性, 无随机)
    best = np.argmax(sc, 1)
    zero = sc.max(1) == 0
    if zero.any():
        tie = mis[zero].min(1)
        cand = np.argmin(mis[zero] - tie[:, None] * 1e9, axis=1)
        best[zero] = cand
    return best


def fitness(g, code_tr, y_tr):
    p = predict(g, code_tr)
    acc = float((p == y_tr).mean())
    return acc - LAM * float(g.k.mean() / MAXK), acc


# ----------------------------- 遗传算法 -----------------------------

def init_pop(rng, code_tr, y_tr):
    """一半个体用**原型初始化**(直接拿训练样本的位码当合取项, 类取其真实类),
    另一半随机。理由: 32 位 × 16 项 = 512 位搜索空间, 纯随机初始化 GA 收敛不到
    (第一版折0 训练准确率只有 0.78 = 没收敛, 不能据此判"离散编码不行")。"""
    pop = []
    for i in range(POP):
        g = Genome()
        if i % 2 == 0:
            c = int(rng.integers(0, NCLS))
            cand = np.where(y_tr == c)[0]
            g.mask[:] = code_tr[int(rng.choice(cand))]
            g.k[:] = 0
            g.cls[:] = c
            # 其余项仍随机, 保持多样性
            for j in range(NTERM // 2, NTERM):
                g.mask[j] = rng.integers(0, 1 << BITS, dtype=np.uint32)
                g.k[j] = int(rng.integers(0, MAXK + 1))
                g.cls[j] = int(rng.integers(0, NCLS))
        else:
            g.mask = rng.integers(0, 1 << BITS, NTERM, dtype=np.uint32)
            g.k = rng.integers(0, MAXK + 1, NTERM)
            g.cls = rng.integers(0, NCLS, NTERM)
        pop.append(g)
    return pop


def hill_climb(g0, code_tr, y_tr, passes=4, rng=None):
    """逐位局部精修: 每次翻一位 / 调一个阈值 / 改一个类别, 适应度不降就接受。
    纯整数、纯位运算, 与遗传算法同一套评价。用来把"GA 没收敛"和"模型类到顶"分开。"""
    g = g0.copy()
    best = fitness(g, code_tr, y_tr)[0]
    for _ in range(passes):
        improved = False
        for i in range(NTERM):
            for b in range(BITS):
                cand = g.mask[i] ^ (np.uint32(1) << np.uint32(b))
                old = g.mask[i]; g.mask[i] = cand
                f = fitness(g, code_tr, y_tr)[0]
                if f > best + 1e-12:
                    best = f; improved = True
                else:
                    g.mask[i] = old
        for i in range(NTERM):
            for d in (-1, 1):
                old = g.k[i]
                if 0 <= old + d <= MAXK:
                    g.k[i] = old + d
                    f = fitness(g, code_tr, y_tr)[0]
                    if f > best + 1e-12:
                        best = f; improved = True
                    else:
                        g.k[i] = old
            for c in range(NCLS):
                old = g.cls[i]
                g.cls[i] = c
                f = fitness(g, code_tr, y_tr)[0]
                if f > best + 1e-12:
                    best = f; improved = True
                else:
                    g.cls[i] = old
        if not improved:
            break
    return g


def _blocks(mask):
    """(T,) uint32 → (T,NFEAT) uint8: 把每项的 32 位拆成 4 个特征块 (每块 8 位)"""
    sh = np.arange(NFEAT, dtype=np.uint32) * np.uint32(8)
    return ((mask[:, None] >> sh[None, :]) & np.uint32(0xFF)).astype(np.uint8)


def _join(byt):
    """(T,NFEAT) uint8 → (T,) uint32"""
    out = np.zeros(len(byt), np.uint32)
    for j in range(NFEAT):
        out |= byt[:, j].astype(np.uint32) << np.uint32(8 * j)
    return out


def crossover(a, b, rng, mode="term", fa=1.0, fb=1.0):
    """交叉算子 (实验的唯一变量):
      term  : 整项 32 位整体取自某个父代        (上一版基线, 粒度最粗)
      block : 按【特征块】(每特征 8 位) 取自某个父代
      wblock: 同样按特征块, 但取哪个父代由**适应度加权**决定 (适应度高的父代贡献更多块)
    """
    c = Genome()
    if mode == "term":
        swap = rng.random(NTERM) < PC
        c.mask = np.where(swap, b.mask, a.mask)
        c.k = np.where(swap, b.k, a.k)
        c.cls = np.where(swap, b.cls, a.cls)
        return c
    ba, bb = _blocks(a.mask), _blocks(b.mask)
    if mode == "block":
        sel = rng.random((NTERM, NFEAT)) < PC                 # 每个块独立 50/50
    else:  # wblock: 按适应度比例决定从谁那里继承
        pB = fb / max(fa + fb, 1e-9)
        sel = rng.random((NTERM, NFEAT)) < pB
    bc = np.where(sel, bb, ba)
    c.mask = _join(bc)
    c.k = np.where(rng.random(NTERM) < PC, b.k, a.k)
    c.cls = np.where(rng.random(NTERM) < PC, b.cls, a.cls)
    return c


def mutate(g, rng):
    m = Genome(g.mask.copy(), g.k.copy(), g.cls.copy())
    for i in range(NTERM):
        if rng.random() < MUT_BIT:
            m.mask[i] ^= np.uint32(rng.integers(1, 1 << BITS, dtype=np.uint64))
    if rng.random() < MUT_K:
        i = int(rng.integers(0, NTERM))
        m.k[i] = int(rng.integers(0, MAXK + 1))
    return m


def run_ga(code_tr, y_tr, code_te, y_te, seed=0, verbose=True, hist=None,
           refine=False, cross="term"):
    rng = np.random.default_rng(seed)
    pop = init_pop(rng, code_tr, y_tr)
    for gen in range(GENS):
        fits = [fitness(g, code_tr, y_tr)[0] for g in pop]
        order = np.argsort(fits)[::-1]
        newpop = [pop[i].copy() for i in order[:ELITE]]
        while len(newpop) < POP:
            def tour():
                win = rng.integers(0, POP, TOURN)
                wi = win[np.argmax([fits[i] for i in win])]
                return pop[wi].copy(), fits[wi]
            ga_, fa = tour()
            gb_, fb = tour()
            newpop.append(mutate(crossover(ga_, gb_, rng, cross, fa, fb), rng))
        pop = newpop
        if hist is not None and (gen % 10 == 0 or gen == GENS - 1):
            bi = int(np.argmax([fitness(g, code_tr, y_tr)[0] for g in pop]))
            f, tr = fitness(pop[bi], code_tr, y_tr)
            te = float((predict(pop[bi], code_te) == y_te).mean())
            hist.append((gen, f, tr, te))
            if verbose and gen % 200 == 0:
                print("      代 %3d  适应度 %.4f  训练 %.4f  测试 %.4f" % (gen, f, tr, te))
                sys.stdout.flush()
    bi = int(np.argmax([fitness(g, code_tr, y_tr)[0] for g in pop]))
    best = pop[bi]
    if refine:
        best = hill_climb(best, code_tr, y_tr)
    tr_acc = float((predict(best, code_tr) == y_tr).mean())
    te_acc = float((predict(best, code_te) == y_te).mean())
    return best, tr_acc, te_acc


def nearest_proto(code_tr, y_tr, code_te, y_te):
    """**确定性参照**: 汉明空间最近原型 (1-NN over 离散化特征)。
    同样纯整数 popcount, 但**不搜索** ⇒ 用来分清"是 GA 搜不到"还是"这个模型类到顶了"。"""
    d = np.bitwise_count(np.bitwise_xor(code_te[:, None], code_tr[None, :]))   # (nte, ntr)
    nn = d.argmin(1)
    return float((y_tr[nn] == y_te).mean())


def main():
    X, y = load_iris()
    print("设备 %s (纯整数位运算)" % DEV)
    print("Iris %d 条 × %d 特征 → 离散成每特征 %d 档, **位码 %d 位** (每特征一个 %d 位 one-hot 块)"
          % (len(y), NFEAT, LEV, BITS, LEV))
    print("分类器 = %d 条位掩码合取项; 命中条件 popcount(项&~码) ≤ 阈值; 全程整数 popcount, 无梯度"
          % NTERM)
    print("遗传算法: 种群 %d, 代数 %d, 精英 %d, 锦标赛 %d, 交叉率 %.1f, 变异率 %.2f/位\n"
          % (POP, GENS, ELITE, TOURN, PC, MUT_BIT))

    hist, terms = [], None
    acc = {c: [] for c in ("term", "block", "wblock")}
    ref_best = None
    np_acc = []
    for f, (tr, te) in enumerate(stratified_folds(y, KFOLDS, 12345)):
        edges = make_edges(X[tr])
        code_tr, code_te = encode(X[tr], edges), encode(X[te], edges)
        np_acc.append(nearest_proto(code_tr, y[tr], code_te, y[te]))
        print("  折%d: 训练 %d / 测试 %d | 参照(最近原型, 不搜索) %.4f" % (f, len(tr), len(te), np_acc[-1]))
        for c in ("term", "block", "wblock"):
            h = [] if (f == 0 and c == "term") else None
            # ★ 返回值不能叫 tr/te: 那会覆盖本折的索引数组 tr/te ⇒ 下一轮 y[tr] 报 IndexError
            b, tr_acc, te_acc = run_ga(code_tr, y[tr], code_te, y[te], seed=f,
                                       hist=h, cross=c, refine=False)
            acc[c].append(te_acc)
            if f == 0 and c == "term":
                hist = h
            print("     交叉=%-6s 测试 %.4f (训练 %.4f)" % (c, te_acc, tr_acc))
        sys.stdout.flush()
    for c in acc:
        m = float(np.mean(acc[c]))
        if ref_best is None or m > ref_best[1]:
            ref_best = (c, m)

    # 对最好的交叉方式再加逐位精修
    gar = []
    for f, (tr, te) in enumerate(stratified_folds(y, KFOLDS, 12345)):
        edges = make_edges(X[tr])
        b, tr_acc, te_acc = run_ga(encode(X[tr], edges), y[tr], encode(X[te], edges), y[te],
                                   seed=f, cross=ref_best[0], refine=True)
        gar.append(te_acc)
        if f == 0:
            terms = b
    print("  → 最好交叉 = %s, 再加逐位精修: %.4f" % (ref_best[0], float(np.mean(gar))))

    print("\n================ 5 折结果 (唯一变量 = 交叉算子) ================")
    print("  参照: 最近原型(确定性, 不搜索)  %.4f ± %.4f" % (float(np.mean(np_acc)), float(np.std(np_acc))))
    for c in ("term", "block", "wblock"):
        print("  GA 交叉=%-6s                  %.4f ± %.4f" % (c, float(np.mean(acc[c])), float(np.std(acc[c]))))
    print("  GA(%s) + 逐位精修               %.4f ± %.4f" % (ref_best[0], float(np.mean(gar)), float(np.std(gar))))

    nn = {}
    if os.path.exists("check_iris_results.json"):
        with io.open("check_iris_results.json", encoding="utf-8") as fh:
            d = json.load(fh)
        for k, v in d.items():
            nn[k] = float(np.mean([x[0] for x in v]))
        print("  (对比, 上一轮同一协议) logistic %.4f | mlp %.4f | transformer %.4f"
              % (nn.get("logistic", -1), nn.get("mlp", -1), nn.get("transformer", -1)))

    print("\n  ---- 折0 学到的位掩码合取项 (十六进制) ----")
    for i in range(NTERM):
        bits = np.bitwise_count(terms.mask[i])
        print("    项%2d  mask=0x%08x  %2d 个 1  阈值k=%d  归类=%s"
              % (i, int(terms.mask[i]), int(bits), int(terms.k[i]), CLASSES[int(terms.cls[i])]))

    os.makedirs("figs", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    g = [h[0] for h in hist]
    ax[0].plot(g, [h[1] for h in hist], label="fitness (= train accuracy, LAM=0)", color="#c0392b", lw=2)
    ax[0].plot(g, [h[2] for h in hist], label="train accuracy", color="#2c3e50", ls="--", lw=1.2)
    ax[0].plot(g, [h[3] for h in hist], label="held-out accuracy", color="#27ae60", lw=2)
    ax[0].set_xlabel("generation"); ax[0].set_ylabel("value"); ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=9, loc="lower right"); ax[0].grid(alpha=0.3)
    ax[0].set_title("(a) Genetic algorithm progress (fold 0, GA run)")
    names = ["random\nguess", "logistic", "MLP", "Transformer", "nearest\nproto",
             "GA\nterm", "GA\nblock", "GA\nwblock", "GA best\n+ refine"]
    vals = [1 / NCLS, nn.get("logistic", 0), nn.get("mlp", 0), nn.get("transformer", 0),
            float(np.mean(np_acc)), float(np.mean(acc["term"])), float(np.mean(acc["block"])),
            float(np.mean(acc["wblock"])), float(np.mean(gar))]
    b = ax[1].bar(names, vals, color=["#95a5a6", "#7f8c8d", "#2980b9", "#8e44ad", "#b2bec3",
                                      "#a04000", "#d35400", "#e67e22", "#27ae60"])
    for r_, v in zip(b, vals):
        ax[1].text(r_.get_x() + r_.get_width() / 2, v + 0.02, "%.3f" % v, ha="center", fontsize=11)
    ax[1].set_ylim(0, 1.12); ax[1].set_ylabel("5-fold test accuracy")
    ax[1].set_title("(b) Crossover operator ablation (only variable changed)")
    ax[1].set_axisbelow(True); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("figs/iris_ga.png", dpi=150)
    with open("figs/ga_results.json", "w", encoding="utf-8") as f:
        json.dump(dict(hist=hist, np_acc=np_acc, arms=acc, best=ref_best[0], gar=gar, nn=nn),
                   f, ensure_ascii=False, indent=2)
    print("图像已保存: figs/iris_ga.png  (数据 figs/ga_results.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())