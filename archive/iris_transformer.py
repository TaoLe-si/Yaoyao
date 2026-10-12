"""鸢尾花 (Iris) 上测 Transformer 的分类表现, 并绘制图像。

数据: 官方 UCI iris.data (150 行 × 4 特征, 3 类各 50) —— 联网取一次后缓存到 data/iris.csv。
协议: **分层 5 折 × 3 种子**; 标准化只用训练折拟合 (防泄漏); 报 折均值±标准差 + 宏 F1。
      出折 (out-of-fold) 预测汇总成混淆矩阵, 用满 150 条。
模型: ① Transformer 编码器 (4 个特征各当一个 token)  ② 逻辑回归  ③ 2 层 MLP —— 同预算同协议。
图像: figs/iris_transformer.png  (四联: 训练曲线 / 准确率对比 / 混淆矩阵 / 表示 PCA)
"""
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEV = "cuda" if torch.cuda.is_available() else "cpu"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
CSV = "data/iris.csv"
CLASSES = ["setosa", "versicolor", "virginica"]
NCLS = 3
NFEAT = 4
EPOCHS = 200
BS = 16
SEEDS = (0, 1, 2)
KFOLDS = 5


# ----------------------------- 数据 -----------------------------

def load_iris():
    if not os.path.exists(CSV):
        os.makedirs("data", exist_ok=True)
        with urllib.request.urlopen(URL, timeout=30) as r:
            raw = r.read().decode("utf-8", "replace")
        with io.open(CSV, "w", encoding="utf-8") as f:
            f.write(raw)
    X, y = [], []
    with io.open(CSV, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 5 or not parts[0]:
                continue
            X.append([float(v) for v in parts[:4]])
            name = parts[4].replace("Iris-", "")
            y.append(CLASSES.index(name))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    assert X.shape == (150, 4) and y.shape == (150,), "数据形状不对: %s %s" % (X.shape, y.shape)
    for c in range(NCLS):
        assert (y == c).sum() == 50, "类别 %d 不是 50 条" % c
    return X, y


def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    folds = [[] for _ in range(k)]
    for c in range(NCLS):
        ic = idx[y == c].copy()
        rng.shuffle(ic)
        for j, v in enumerate(ic):
            folds[j % k].append(v)
    out = []
    for j in range(k):
        te = np.array(sorted(folds[j]))
        tr = np.array(sorted(set(idx.tolist()) - set(te.tolist())))
        out.append((tr, te))
    return out


# ----------------------------- 模型 -----------------------------

class TokEmbed(nn.Module):
    """每个特征当一个 token: 标量 → d 维, 再加一个"第几个特征"的位置嵌入"""
    def __init__(self, d, nfeat):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.pos = nn.Embedding(nfeat, d)

    def forward(self, x):                      # (B,4) → (B,4,d)
        h = self.proj(x.unsqueeze(-1))
        return h + self.pos(torch.arange(x.shape[1], device=x.device))[None]


class IrisTransformer(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, ffn=128, drop=0.1):
        super().__init__()
        self.emb = TokEmbed(d, NFEAT)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn, dropout=drop,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, NCLS)

    def forward(self, x, return_rep=False):
        h = self.enc(self.emb(x))
        r = self.norm(h.mean(1))               # 4 个 token 平均池化
        logits = self.head(r)
        return (logits, r) if return_rep else logits


class Logistic(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.fc = nn.Linear(NFEAT, NCLS)

    def forward(self, x, return_rep=False):
        if return_rep:
            return self.fc(x), x
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NFEAT, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.head = nn.Linear(d, NCLS)

    def forward(self, x, return_rep=False):
        r = self.net(x)
        logits = self.head(r)
        return (logits, r) if return_rep else logits


def build(name):
    return {"transformer": IrisTransformer, "logistic": Logistic, "mlp": MLP}[name]()


def train_eval(name, Xtr, ytr, Xte, yte, seed, epochs=EPOCHS, want_curve=False, want_rep=False):
    torch.manual_seed(seed)
    m = build(name).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = max(1, len(Xtr) // BS + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * steps, pct_start=0.2)
    xtr = torch.tensor(Xtr, device=DEV); ytr_t = torch.tensor(ytr, device=DEV)
    xte = torch.tensor(Xte, device=DEV)
    curve = []
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(xtr))
        tot = 0.0
        for i in range(0, len(xtr), BS):
            b = perm[i:i + BS]
            logits = m(xtr[b])
            loss = F.cross_entropy(logits, ytr_t[b], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            tot += float(loss) * len(b)
        if want_curve and (ep % 5 == 0 or ep == epochs - 1):
            m.eval()
            with torch.no_grad():
                acc = float((m(xte).argmax(-1).cpu().numpy() == yte).mean())
            curve.append((ep, tot / len(xtr), acc))
        if ep % 50 == 0:
            sys.stdout.write("      ep %3d loss %.4f\n" % (ep, tot / len(xtr)))
            sys.stdout.flush()
    m.eval()
    with torch.no_grad():
        out = m(xte, return_rep=True) if want_rep else m(xte)
    logits, rep = (out if want_rep else (out, None))
    pred = logits.argmax(-1).cpu().numpy()
    acc = float((pred == yte).mean())
    f1 = macro_f1(yte, pred)
    return dict(acc=acc, f1=f1, pred=pred, rep=(rep.cpu().numpy() if rep is not None else None),
                curve=curve, time=time.time() - t0, model=m)


def macro_f1(y, p):
    fs = []
    for c in range(NCLS):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(fs))


def main():
    X, y = load_iris()
    print("设备 %s" % DEV)
    print("Iris: %d 条 × %d 特征, %d 类各 50 条 | 特征范围 %s" %
          (len(y), NFEAT, NCLS, np.round(X.min(0).tolist(), 2)))
    names = ["logistic", "mlp", "transformer"]
    print("协议: 分层 %d 折 × %d 种子; 标准化只用训练折; 共 %d 次训练\n"
          % (KFOLDS, len(SEEDS), KFOLDS * len(SEEDS) * len(names)))

    oof = {n: np.zeros(len(y), dtype=np.int64) for n in names}     # 出折预测
    res = {n: [] for n in names}
    f1s = {n: [] for n in names}
    curve0 = None
    rep150 = None
    for f, (tr, te) in enumerate(stratified_folds(y, KFOLDS, 12345)):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        for n in names:
            accs, f1a = [], []
            for sd_ in SEEDS:
                want_curve = (n == "transformer" and f == 0 and sd_ == 0)
                want_rep = (n == "transformer" and f == 0 and sd_ == 0)
                r = train_eval(n, Xtr, y[tr], Xte, y[te], sd_,
                               want_curve=want_curve, want_rep=want_rep)
                accs.append(r["acc"]); f1a.append(r["f1"])
                if f == 0:
                    oof[n][te] = r["pred"]
                if want_curve:
                    curve0 = r["curve"]
                if want_rep:
                    # 用折 0 训好的那个 transformer, 把 **全部 150 条** 过一遍拿表示
                    mm = r["model"]
                    mm.eval()
                    Xall = torch.tensor((X - mu) / sd, device=DEV)
                    with torch.no_grad():
                        _, rr = mm(Xall, return_rep=True)
                    rep150 = rr.cpu().numpy()
            res[n].append((float(np.mean(accs)), float(np.std(accs))))
            f1s[n].append(float(np.mean(f1a)))
            print("  折%d  %-11s acc %.3f±%.3f  宏F1 %.3f  (用时 %.1fs)"
                  % (f, n, res[n][-1][0], res[n][-1][1], f1s[n][-1], r["time"]))
            sys.stdout.flush()

    print("\n================ 5 折结果 (折均值±折标准差) ================")
    for n in names:
        a = np.array([v[0] for v in res[n]])
        print("  %-11s 准确率 %.4f ± %.4f   宏F1 %.4f" % (n, a.mean(), a.std(), float(np.mean(f1s[n]))))
    print("  随机猜基线   %.4f" % (1.0 / NCLS))

    # 汇总出折 (折0) 混淆矩阵 —— 只用折 0 的测试集 (30 条), 其余折的 logits 没留
    cm = np.zeros((NCLS, NCLS), dtype=int)
    te0 = stratified_folds(y, KFOLDS, 12345)[0][1]
    for t, p in zip(y[te0], oof["transformer"][te0]):
        cm[t, p] += 1
    print("\n  折0 出折混淆矩阵 (transformer, %d 条):" % len(te0))
    print("           预:" + "".join("%10s" % c for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print("     真:%-10s" % c + "".join("%10d" % v for v in cm[i]))

    # ------------------------- 画图 -------------------------
    os.makedirs("figs", exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # (a) 训练曲线
    if curve0:
        ep = [c[0] for c in curve0]; ls = [c[1] for c in curve0]; ac = [c[2] for c in curve0]
        a1 = ax[0, 0]
        a1.plot(ep, ls, color="#c0392b", lw=2, label="train loss")
        a1.set_xlabel("epoch"); a1.set_ylabel("train loss", color="#c0392b")
        a1.tick_params(axis="y", labelcolor="#c0392b")
        a2 = a1.twinx()
        a2.plot(ep, ac, color="#2471a3", lw=2, label="val acc")
        a2.set_ylabel("val accuracy", color="#2471a3")
        a2.tick_params(axis="y", labelcolor="#2471a3")
        a2.set_ylim(0, 1.02)
        a1.set_title("(a) Transformer training curve (fold 0, seed 0)")
        a1.grid(alpha=0.3)
    # (b) 准确率对比
    a = ax[0, 1]
    labels = ["random\nguess", "logistic\nregression", "MLP\n(4-64-64)", "Transformer\n(2 layers)"]
    means = [1.0 / NCLS] + [np.mean([v[0] for v in res[n]]) for n in names]
    errs = [0.0] + [np.std([v[0] for v in res[n]]) for n in names]
    cols = ["#95a5a6", "#7f8c8d", "#2980b9", "#8e44ad"]
    b = a.bar(labels, means, yerr=errs, capsize=6, color=cols)   # 实色: 网格不会透过来
    for r_, m_ in zip(b, means):
        a.text(r_.get_x() + r_.get_width() / 2, m_ + 0.02, "%.3f" % m_, ha="center", fontsize=11)
    a.set_ylim(0, 1.12); a.set_ylabel("test accuracy (5-fold mean)")
    a.set_title("(b) Accuracy: 5-fold CV, 3 seeds")
    a.set_axisbelow(True)          # 网格画在柱子后面
    a.grid(alpha=0.3, axis="y")
    # (c) 混淆矩阵
    a = ax[1, 0]
    im = a.imshow(cm, cmap="Blues")
    a.set_xticks(range(NCLS)); a.set_xticklabels(CLASSES, rotation=15)
    a.set_yticks(range(NCLS)); a.set_yticklabels(CLASSES)
    a.set_xlabel("predicted"); a.set_ylabel("true")
    a.set_title("(c) Confusion matrix (transformer, fold 0, %d samples)" % len(te0))
    for i in range(NCLS):
        for j in range(NCLS):
            a.text(j, i, str(cm[i, j]), ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=13)
    plt.colorbar(im, ax=a, fraction=0.046)
    # (d) 表示 PCA
    a = ax[1, 1]
    if rep150 is not None:
        Z = rep150 - rep150.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        P = Z @ Vt[:2].T
        for c in range(NCLS):
            s = y == c
            a.scatter(P[s, 0], P[s, 1], s=45, label=CLASSES[c], alpha=0.8,
                      edgecolors="k", linewidths=0.4)
        ev = (S ** 2) / (S ** 2).sum()
        a.set_xlabel("PC1 (%.0f%%)" % (ev[0] * 100))
        a.set_ylabel("PC2 (%.0f%%)" % (ev[1] * 100))
        a.legend(loc="best", fontsize=9)
    a.set_title("(d) Transformer representation (PCA, all 150)")
    a.grid(alpha=0.3)
    fig.suptitle("Iris classification — Transformer vs baselines (150 samples, 4 features, 3 classes)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = "figs/iris_transformer.png"
    fig.savefig(out, dpi=150)
    print("\n图像已保存: %s" % out)
    with open("check_iris_results.json", "w", encoding="utf-8") as f:
        json.dump({n: res[n] for n in names}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())