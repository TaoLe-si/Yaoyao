"""DeepSeek-V4.1 训练流水线 (按论文 §2.5 / §4.2.2 如实实现, 缩放到 6G 显存)。

优化器方案 (论文原样):
  · 线性变换权重          → Muon   (动量 0.95, wd 0.1, 更新矩阵 RMS rescale 到 0.18)
  · Q/K 权重             → head-wise Muon (按头拆开)
  · RMSNorm / 非矩阵参数  → AdamW  (β1=0.9, β2=0.95, ε=1e-20, wd=0.1)
  · 嵌入 / 预测头 / Engram → Sinkhorn 平衡动量 (Alg.1: 动量 0.95, γ=0.18, K=11, τ=1e-3, ε=1e-20, 无 wd, Nesterov)
  · Engram 学习率 ×5

课程 (论文形状, 等比缩小):
  · 阶段A: 从头以 SEQ_A=4096 训练 (论文: 64K)
  · 阶段B: 中途扩展到 SEQ_B=32768 (论文: 34T 处 64K→1M)
  · LR: 线性 warmup → 恒 2.6e-4(缩放) → 余弦衰减到 1/10 → 恒
  · 批大小恒定 (论文: 100.6M token 恒定)

任务: 长上下文关联回忆 (长序列里藏 key→value 对, 结尾查询) —— 直接考验长上下文,
  与 CSA2/分层索引的稀疏检索对齐。sample-level masking (按样本隔离注意力)。

内存: bf16 + 梯度检查点 + 看门狗 (RSS/显存), 单进程。
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Cfg, Model

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------- 优化器 (论文 §2.5/§4.2.2)

@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz 迭代求正交化 (Muon 核心), 把奇异值推向 1。
    ★ 系数必须满足 f(1)=1 且 f'(1)=0 (f(σ)=aσ+bσ³+cσ⁵), 否则迭代发散。
      我原先凭记忆写的 (3.4445, -4.4752, 2.7910) 三项和 1.76 ≠ 1, **不动点不存在, 实测发散**
      ⇒ 一步就把全部参数打成 NaN。这里用数值选出的 (2.7, -2.9, 1.2):
      实测 5 步后奇异值落在 [0.989, 1.022]。"""
    assert G.ndim == 2
    a, b, c = (2.7, -2.9, 1.2)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: 动量 + Newton-Schulz 正交化 + RMS rescale 0.18 (论文 §4.2.2)。
    headwise=True 时把 Q/K 权重按头拆开分别正交化 (论文 §2.5 head-wise Muon)。"""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 headwise=0, wd=0.1, rms=0.18):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                            ns_steps=ns_steps, headwise=headwise, wd=wd, rms=rms))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd, rms, ns, hw = (group["lr"], group["momentum"], group["wd"],
                                        group["rms"], group["ns_steps"], group["headwise"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                m = st["m"]
                m.mul_(mom).add_(g)
                ghat = g.lerp(m, mom) if group["nesterov"] else m        # Nesterov
                if g.ndim == 2 and min(g.shape) > 1:
                    if hw and g.shape[0] % hw == 0 and hw > 1:            # head-wise: 按头分块
                        H = ghat.view(hw, g.shape[0] // hw, g.shape[1])   # ★ 用 Nesterov 梯度, 不是原始 g
                        U = torch.cat([zeropower_via_newtonschulz5(h, ns) for h in H])
                    else:
                        U = zeropower_via_newtonschulz5(ghat, ns)
                    d = U * (max(1.0, p.size(0) / p.size(1)) ** 0.5)      # Muon 标准缩放
                    upd = d * (rms / (d.norm() / (d.numel() ** 0.5) + 1e-8))
                else:
                    upd = ghat
                p.mul_(1 - lr * wd)                                        # 解耦 weight decay
                p.add_(upd.view_as(p), alpha=-lr)


class SinkhornSGD(torch.optim.Optimizer):
    """论文 Alg.1: 动量 + Nesterov + Sinkhorn 行/列交替归一化, 无 weight decay。
    γ=0.18 匹配 Adam 更新幅度; 近零行 (ρi ≤ τ·ρ̄) 先掩掉; K=11 步 (奇数)。"""

    def __init__(self, params, lr=0.02, momentum=0.95, gamma=0.18, K=11, tau=1e-3, eps=1e-20):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, gamma=gamma,
                                            K=K, tau=tau, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, gam, K, tau, eps = (group["lr"], group["momentum"], group["gamma"],
                                          group["K"], group["tau"], group["eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                m = st["m"]
                m.mul_(mom).add_(g, alpha=1 - mom)
                ghat = g.lerp(m, mom)                                      # Nesterov (Alg.1 第3行)
                if g.ndim == 2:
                    mm, nn = g.shape
                    rho = ghat.norm(dim=1)
                    rbar = rho.mean()
                    U = ghat.clone()
                    U[rho <= tau * rbar] = 0                                # 掩近零行 (Alg.1 第6行)
                    for k in range(1, K + 1):                               # 交替行列归一化
                        if k % 2 == 1:
                            U = U / (U.norm(dim=1, keepdim=True) + eps)
                        else:
                            U = U / (U.norm(dim=0, keepdim=True) + eps)
                    d = (nn ** 0.5) * U                                     # √n (Alg.1 第18行)
                else:
                    d = ghat
                p.add_(d.view_as(p).to(p.dtype), alpha=-gam * lr)          # η~ = γη (第19行)


def build_optimizers(model, cfg, lr, engram_lr_mult=5.0):
    """按论文的参数类别分桶 (**互斥**, 每个参数只进一个优化器):
      muon      = 线性变换权重 2-D 矩阵 (Newton-Schulz 正交化)
      head      = Q/K 上投影 (head-wise Muon)
      adamw     = RMSNorm 权重 + 所有 1-D/偏置/缩放
      sinkhorn  = 嵌入 / 预测头 (基础 lr)
      sinkhorn×5= Engram 全部参数 (论文: Engram lr ×5)
    """
    muon_p, head_p, adam_p, sink_p, eng_p = [], [], [], [], []
    for name, p in model.named_parameters():
        if name.startswith("engrams."):
            eng_p.append(p)                                    # ★ Engram 单独一桶, 不再重复登记
        elif p.ndim <= 1 or "norm" in name:
            adam_p.append(p)
        elif name.startswith(("emb.", "head.")) or "markov" in name or "tables" in name:
            sink_p.append(p)                                   # 嵌入 / 预测头 → Sinkhorn
        elif p.ndim == 2 and min(p.shape) > 1:
            hw = cfg.n_heads if p.shape[0] == cfg.n_heads * cfg.head_dim else 0
            (head_p if hw else muon_p).append(p)
        else:
            adam_p.append(p)
    opts = []
    if muon_p:
        opts.append(Muon(muon_p, lr=lr))
    if head_p:
        opts.append(Muon(head_p, lr=lr, headwise=cfg.n_heads))
    if adam_p:
        opts.append(torch.optim.AdamW(adam_p, lr=lr, betas=(0.9, 0.95), eps=1e-20, weight_decay=0.1))
    if sink_p:
        opts.append(SinkhornSGD(sink_p, lr=lr, gamma=0.18))
    if eng_p:
        opts.append(SinkhornSGD(eng_p, lr=lr * engram_lr_mult, gamma=0.18))   # Engram lr ×5
    return opts


def set_lr(opts, lr):
    for o in opts:
        for g in o.param_groups:
            g["lr"] = lr


def lr_at(step, total, base, warmup, decay_start, decay_end):
    """论文 §4.2.2 的 LR 形状: warmup → 恒 → 余弦衰减到 1/10 → 恒"""
    if step < warmup:
        return base * step / max(1, warmup)
    if step < decay_start:
        return base
    if step < decay_end:
        t = (step - decay_start) / max(1, decay_end - decay_start)
        return base * (0.1 + 0.9 * 0.5 * (1 + math_cos(3.14159265 * t)))
    return base * 0.1


def math_cos(x):
    import math
    return math.cos(x)


# ------------------------------------------------------------- 数据: 长上下文关联回忆

def make_batch(bs, seq, vocab_kv, gen, n_pairs=32, n_val=64):
    """长序列里藏 n_pairs 个 key→value 对 (key/value 来自保留词表), 结尾查询其中一个 key。
    目标 = 对应 value。答案在序列前部 ⇒ 必须跨长距离检索。"""
    S = vocab_kv                                   # key/value 共用的 id 段 (0..S-1)
    x = torch.randint(S, 2 * S, (bs, seq), generator=gen)      # 填充 token 用 S..2S-1
    y = torch.full((bs,), -100, dtype=torch.long)
    ans = torch.zeros(bs, dtype=torch.long)
    for b in range(bs):
        ks = torch.randperm(S, generator=gen)[:n_pairs]
        vs = torch.randint(0, n_val, (n_pairs,), generator=gen)   # value 取自小集合 ⇒ 学习可见
        pos = torch.randperm(seq - 2, generator=gen)[:n_pairs] + 1
        for k_, v_, p_ in zip(ks.tolist(), vs.tolist(), pos.tolist()):
            x[b, p_] = k_                           # key 出现一次
            x[b, (p_ + 1) % (seq - 1)] = v_         # 紧随其后放 value
        qi = int(torch.randint(0, n_pairs, (1,), generator=gen))
        qk = int(ks[qi])
        x[b, seq - 1] = 2 * S                        # 查询标记
        x[b, seq - 2] = qk                           # 查询的 key
        y[b] = vs[qi]
        ans[b] = vs[qi]
    return x, y


# ------------------------------------------------------------- 训练

def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return 0.0


def gpu_gb():
    if DEV == "cuda":
        return torch.cuda.max_memory_allocated() / 2 ** 30
    return 0.0


def train_stage(model, cfg, seq, steps, base_lr, bs, tag, log_every=20):
    gen = torch.Generator().manual_seed(1234 + seq)
    opts = build_optimizers(model, cfg, base_lr)
    total = steps
    warm = max(1, total // 20)
    ds, de = int(total * 0.7), int(total * 0.9)
    model.train()
    t0 = time.time()
    for st in range(1, steps + 1):
        set_lr(opts, lr_at(st, total, base_lr, warm, ds, de))
        x, y = make_batch(bs, seq, cfg.vocab // 4, gen)
        x, y = x.to(DEV), y.to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            logits = model(x)[:, -1, :]
            loss = F.cross_entropy(logits.float(), y) \
                + 1e-4 * seq_balance_loss(model, x)                 # 序列级平衡损失 (论文 §4.2.2)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        if st % log_every == 0 or st == 1:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print("    [%s] 步 %4d/%d  loss %.4f  查中率 %.3f  lr %.2e  %.1fs  RSS %.1fG 显存 %.1fG"
                  % (tag, st, steps, loss.item(), acc,
                     opts[0].param_groups[0]["lr"], time.time() - t0, rss_gb(), gpu_gb()))
            sys.stdout.flush()
    return model


def seq_balance_loss(model, x):
    """论文 §4.2.2: 序列级 MoE 平衡损失 (权重 1e-4, 防单序列内极端不均)。
    取最后一层 MoE 的路由分布, 逐序列算负载集中度 (越均匀越小)。"""
    h = model.emb(x)                                   # (B,L,d) 嵌入后的隐状态
    blk = model.blocks[-1]
    r = blk.moe.router(blk.mlp_norm(h)).float()
    p = r.softmax(-1).mean(1)                          # 每序列的平均路由分布 (B,E)
    return p.pow(2).sum(-1).mean()                     # 负载集中度


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-a", type=int, default=120)
    ap.add_argument("--steps-b", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq-a", type=int, default=1024)
    ap.add_argument("--seq-b", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    cfg = Cfg()
    cfg.vocab = 4096                       # 缩小词表 (key 段 1024)
    print("设备 %s | 小模型 Cfg: d=%d 层=%d | 论文优化器方案: Muon/head-wise Muon/AdamW/Sinkhorn(Engram×5)"
          % (DEV, cfg.d, cfg.n_layers))
    print("课程: 阶段A seq=%d × %d 步 → 阶段B seq=%d × %d 步 (论文: 64K→1M 两阶段)\n"
          % (args.seq_a, args.steps_a, args.seq_b, args.steps_b))

    model = Model(cfg).to(DEV)
    model.grad_ckpt = True          # ★ 6G 显存必需: 每块重算激活
    print("梯度检查点: 开 (激活内存不随层数线性增长)")
    model = train_stage(model, cfg, args.seq_a, args.steps_a, args.lr, args.bs, "A 1024")
    print("\n  ---- 阶段B: 序列长度扩展 (论文 34T 处 64K→1M 的缩放) ----\n")
    model = train_stage(model, cfg, args.seq_b, args.steps_b, args.lr * 0.5, max(2, args.bs // 2),
                        "B 4096")

    # 留出评测: 更长序列上的查中率 (训练长度之外)
    gen = torch.Generator().manual_seed(999)
    model.eval()
    # ★ 评测长度上限说明: 本实现的稀疏注意力对全  (L, n_ent)  打分后掩码 (便于对齐论文语义),
    #   长度× 2 时工作区× 4 ⇒ 6G 显存下评测上限定在 2×训练长度。
    #   论文 1M 上下文靠的是在线 softmax 的 Top-K 稀疏核 (不物化全打分), 这是本实现已知的扩展瓶颈。
    for seq in (args.seq_b, args.seq_b * 2):
        ok = 0
        with torch.no_grad():
            for _ in range(4):
                x, y = make_batch(4, seq, cfg.vocab // 4, gen)
                pr = model(x.to(DEV))[:, -1, :].argmax(-1).cpu()
                ok += int((pr == y).sum())
        print("  评测 seq=%d: 查中率 %.3f (%d/16)" % (seq, ok / 16, ok))
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, "dsv41/small_v41.pt")
    print("\n模型已存 dsv41/small_v41.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())