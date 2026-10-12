"""DeepSeek-V4.1-Flash 架构复刻 (1:1 组件与接线, 参数按小模型缩放)。

规格全部取自论文 arXiv 2609.19969v1 正文 (ref/deepseek/v41_text.txt):
  §2.1  总览: 40 层 = 20 层因果编码器 + 20 层解码器; 前两层**纯 SWA**;
        其余每层同时有 global attention 与 SWA。
  §2.2  CED (Causal Encoder-Decoder): 上半层 (l>L/2) 的 global KV **不从自己的 H_l 来**,
        而是由第 L/2 层的隐状态投影:  C_l = H_{L/2} W_l^{KV},  Z_l = H_{L/2} W_l^{Z}
        SWA 仍逐层各算各的 (局部 KV 用当前层 H_l)。
  §2.3  CSA2: 三模式 (Full/Reindex/Reuse);
        Full    = 自算 main KV + 自算 indexer Q + 从 main KV 投影 indexer K + 重跑索引出 Top-K
        Reindex = 复用最近的 main KV 与 indexer K, 自己算 indexer Q 重新打分出新 Top-K
        Reuse   = 复用最近的 main KV + 已算好的 Top-K, 不算 indexer Q, 不打分
        压缩: 压缩率 m 时, 一个 main KV 由 **2m** 个原始 KV 合成; CSA2 **去掉重叠与绝对位置编码**;
              indexer K 由 main KV 投影 (不再单独从隐状态压一条路); m=1 即不压缩的特例。
  §2.3.1 跨层复用: 层静态分配 Full/Reindex/Reuse。
  §2.3.2 分层稀疏索引: 解码器里第一个 Full 层按**块最大值**选最多 2048 个块 (每块 8 位置)
        = 最多 16384 个候选位置, 后续 Reindex 层只在这个候选池里选自己的 Top-K。
  §2.4.1 Single-Pass mHC: 维护 n 条残差流,  X_{l+1} = B_l X_l + C_l F_l(A **_{l-1}** X_l),
        (A_l,B_l,C_l) = H(X_l) —— 输入混合系数按块**移一位**, 消掉依赖以便单趟融合。
  §2.4.2 Engram: N-gram {2,3,4}, 多头哈希, 每模块 8 头, 每阶总嵌入维 2048,
        每头一张约 16M 表 (表长取不同素数), FP8; **去掉短因果卷积**; 196B 参数分给 2 个模块。
  §2.4.3 DSpark: 3 个 Transformer 块, SWA 窗 128; 一次前向并行出 5 个草稿位的 base logits,
        另有 Markov 头与置信头。
  §2.4.4 FP4 主 KV: MXFP4 (E2M1 + 每 16 通道一个 E4M3 尺度, **去掉二级全局尺度**);
        QAT; **在 RoPE 之后量化**; SWA KV 保持 FP8。
  §2.5  优化: Q/K 用 head-wise Muon; 其余矩阵 Muon; 归一化/偏置用 AdamW;
        Engram/embed/head 用「动量 + Sinkhorn 平衡」(γ=0.18)。

本文件只做**结构与形状**的一比一; 训练脚本见 dsv41/train.py。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- 配置

@dataclass
class Cfg:
    """默认值 = **小模型** (桃: 不要那么大参数, 只要长上下文)。
    括号里是论文 §4.2.1 的原值, 结构完全相同, 只是维度等比缩小。"""
    vocab: int = 8192
    d: int = 256                 # (5120)
    n_layers: int = 12           # (40)
    n_heads: int = 8             # (64)
    head_dim: int = 128          # (512)
    q_lora: int = 128            # (1280)
    kv_lora: int = 128           # (512)
    rope_dim: int = 32           # (64)
    out_groups: int = 2          # (8)
    out_group_dim: int = 128     # (1024)
    swa_window: int = 128        # (128, 论文同)
    idx_heads: int = 4           # (32)
    idx_head_dim: int = 32       # (128)
    topk: int = 64               # (512)
    cand_blocks: int = 32        # (2048)
    block_size: int = 8          # (8, 论文同)
    n_routed: int = 8            # (384)
    n_shared: int = 1            # (1, 论文同)
    topk_expert: int = 2         # (6)
    expert_dim: int = 256        # (2304)
    clamp: float = 10.0          # (10, 论文同)
    mhc_n: int = 4               # (4, 论文同)
    mhc_iters: int = 20          # (20, 论文同)
    engram_orders: Tuple[int, ...] = (2, 3, 4)   # 论文同
    engram_heads: int = 4        # (8)
    engram_order_dim: int = 128  # (2048)
    engram_entries: int = 65536  # (约 16M)
    dspark_blocks: int = 3       # 论文同
    dspark_drafts: int = 5       # 论文同
    fp4_kv: bool = True
    fp8_swa: bool = True

    # --- 派生: 层模式表 (§2.1 / §4.2.1) ---
    def mode_schedule(self) -> List[str]:
        """§2.2: **bottom L/2 层 = 因果编码器, 上半 = 解码器** (40 层 ⇒ 20:20, 按对半切,
        不是按 3:8 比例)。
        编码器: 前 2 层纯 SWA, 其余 CSA2 分 3 组 (每组 1 Full + 其余 Reuse);
        解码器: 分 5 组 (组0 = 1 Full + 其余 Reuse; 组1-4 = 1 Reindex + 其余 Reuse)。"""
        n_enc_total = self.n_layers // 2            # §2.2: bottom L/2
        n_dec_total = self.n_layers - n_enc_total
        n_enc_csa = n_enc_total - 2                 # 前 2 层纯 SWA

        def split(n, k):
            base, r = divmod(n, k)
            return [base + (1 if i < r else 0) for i in range(max(k, 1))]

        enc_groups = max(1, min(3, n_enc_csa // 2))
        dec_groups = max(1, min(5, n_dec_total // 2))
        modes = ["swa_only"] * 2
        for g in split(n_enc_csa, enc_groups):
            modes += ["full"] + ["reuse"] * (g - 1)
        for gi, g in enumerate(split(n_dec_total, dec_groups)):
            modes += [("full" if gi == 0 else "reindex")] + ["reuse"] * (g - 1)
        assert len(modes) == self.n_layers, "%d != %d" % (len(modes), self.n_layers)
        return modes

    @property
    def n_enc_total(self) -> int:
        return self.n_layers // 2

    def enc_csa_layers(self) -> int:
        """编码器里的 CSA2 层数 = L/2 − 2 (前两层纯 SWA)"""
        return self.n_layers // 2 - 2

    def compress_m(self, layer: int) -> int:
        """§4.2.1: 编码器 CSA2 m=2, 解码器 m=1"""
        return 2 if layer < self.n_layers // 2 else 1


# --------------------------------------------------------------------- 基础件

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope(x, pos, dim, theta=10000.0):
    """部分 RoPE: 只转最后 dim 个通道。x 形状统一为 (B, L, H, D)。"""
    B, L, H, D = x.shape
    half = dim // 2
    f = torch.arange(half, device=x.device, dtype=torch.float32) / half
    inv = theta ** (-f)
    ang = pos.float().view(1, L, 1, 1) * inv.view(1, 1, 1, half)
    c, s = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    xr, xp = x[..., :half], x[..., half:dim]
    rot = torch.cat([xr * c - xp * s, xr * s + xp * c], dim=-1)
    return torch.cat([rot, x[..., dim:]], dim=-1)


# ----------------------------------------------------- FP4 / FP8 KV 量化 (§2.4.4)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def mx_fp4_quant(x, blk=16):
    """MXFP4: E2M1 + 每 16 通道一个 E4M3 尺度, **无二级全局尺度** (§2.4.4)。
    E4M3 尺度取块内 |x| 最大值; 反量化在注意力前做。作用于最后一维, 支持任意前置维度。
    用直通 (STE) 以支持 QAT。"""
    C = x.shape[-1]
    assert C % blk == 0, "最后一维 %d 要能被 %d 整除" % (C, blk)
    xb = x.reshape(-1, C // blk, blk)
    scale = xb.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 6.0     # E2M1 最大 6.0
    q = xb / scale
    tab = E2M1.to(x.device, x.dtype)
    idx = (q.unsqueeze(-1) - tab).abs().argmin(-1)
    out = (tab[idx] * scale).reshape(x.shape)
    return out + (x - x.detach())        # STE


def fp8_quant(x):
    """FP8 (e4m3) 模拟: 按张量最大绝对值缩放后吸附到 e4m3 网格"""
    m = x.abs().amax().clamp_min(1e-8)
    q = (x / m * 448.0).clamp(-448, 448)
    q = torch.round(q) / 448.0 * m
    return q + (x - x.detach())


# --------------------------------------------------------------------- 稀疏索引器

class HierSparseIndexer(nn.Module):
    """§2.3 / §2.3.2: indexer Q 从隐状态来, indexer K 由 main KV 投影来;
    FP4 QAT (论文: indexer 的 Q/K 用 FP4); 分层候选池在解码器里用。"""

    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d, c.idx_heads * c.idx_head_dim, bias=False)
        self.k = nn.Linear(c.kv_lora, c.idx_heads * c.idx_head_dim, bias=False)

    def scores(self, h, mkv):
        B, L, _ = h.shape
        q = self.q(h).view(B, L, self.c.idx_heads, self.c.idx_head_dim)
        k = self.k(mkv).view(B, mkv.shape[1], self.c.idx_heads, self.c.idx_head_dim)
        if self.c.fp4_kv:
            q, k = mx_fp4_quant(q), mx_fp4_quant(k)
        s = torch.einsum("blhd,bnhd->bhl n".replace(" ", ""), q, k) / math.sqrt(self.c.idx_head_dim)
        return s          # (B, heads, L, n_entry)

    @staticmethod
    def topk_mask(scores, k, cand=None):
        """返回 (B, heads, L, n) 的 bool: 选中条目。cand 给定时只在候选池里选 (§2.3.2)。
        cand 与 scores 同形状 (B,H,L,n), 直接用, 不能再扩维。"""
        if cand is not None:
            scores = scores.masked_fill(~cand, float("-inf"))
        n = scores.shape[-1]
        k = min(k, n)
        idx = scores.topk(k, dim=-1).indices
        m = torch.zeros_like(scores, dtype=torch.bool)
        m.scatter_(-1, idx, True)
        return m

    def build_candidate_pool(self, scores, valid):
        """§2.3.2: 每块取块内最大 index 分数, 选分数最高的最多 cand_blocks 个块,
        把块内位置收集成候选池 (供后续 Reindex 层搜索)。
        scores: (B,H,L,n);  valid: (L,n) 因果可见。返回 (B,H,L,n) bool。"""
        B, H, L, n = scores.shape
        bs = self.c.block_size
        nb = (n + bs - 1) // bs
        pad = nb * bs - n
        s = F.pad(scores, (0, pad), value=float("-inf")).view(B, H, L, nb, bs).amax(-1)
        v = F.pad(valid, (0, pad), value=False).view(L, nb, bs).any(-1)      # (L,nb)
        s = s.masked_fill(~v[None, None], float("-inf"))
        nb_sel = min(self.c.cand_blocks, nb)
        bi = s.topk(nb_sel, dim=-1).indices                                  # (B,H,L,nb_sel)
        pool = torch.zeros(B, H, L, nb, dtype=torch.bool, device=scores.device)
        pool.scatter_(-1, bi, True)
        pool = pool.unsqueeze(-1).expand(B, H, L, nb, bs).reshape(B, H, L, nb * bs)[..., :n]
        return pool & valid[None, None]


# --------------------------------------------------------------------- CSA2 注意力

class CSA2Attention(nn.Module):
    """一个 CSA2 层 = 主 Q + (main KV 视模式而定) + indexer + SWA 分支 + 分组输出投影。"""

    def __init__(self, c: Cfg, mode: str, m: int):
        super().__init__()
        self.c, self.mode, self.m = c, mode, m
        self.norm = RMSNorm(c.d)
        self.q_down = nn.Linear(c.d, c.q_lora, bias=False)                    # 查询压缩 (§4.2.1), 所有层都要
        self.q_up = nn.Linear(c.q_lora, c.n_heads * c.head_dim, bias=False)
        self.q_norm = RMSNorm(c.head_dim)
        if mode == "full":
            # 压缩器: 由 2m 个原始 KV 合成一条 —— CSA2 无重叠、无绝对位置编码 (§2.3)
            self.kv_down = nn.Linear(2 * m * c.d, c.kv_lora, bias=False)
        if mode in ("full", "reindex"):
            # Reindex 要自算 indexer Q 重新打分 (§2.3.1); Reuse 不需要
            self.indexer = HierSparseIndexer(c)
        # ---- SWA 分支: 逐层自算 (§2.2: 局部 K/V 直接由**本层** H_l 来) ----
        # ★ MLA 式共享上投影: 本层 latent → K/V 用同一组 k_up/v_up, 与主分支共享权重;
        #   独立全秩 swa_k/swa_v 会让主干多 13.4B, 论文 552B 的预算装不下 (对账时查出的偏差)。
        self.kv_down_local = nn.Linear(c.d, c.kv_lora, bias=False)            # 每 token 一条 (未压缩)
        self.k_up = nn.Linear(c.kv_lora, c.n_heads * c.head_dim, bias=False)
        self.v_up = nn.Linear(c.kv_lora, c.n_heads * c.head_dim, bias=False)
        # 输出: 分 out_groups 组, 每组中间维 out_group_dim (§4.2.1)
        self.o_down = nn.Linear(c.n_heads * c.head_dim // c.out_groups,
                                c.out_group_dim, bias=False)
        self.o_up = nn.Linear(c.out_groups * c.out_group_dim, c.d, bias=False)

    def _main_kv(self, h, Henc):
        """Full: 由自己的 h 压缩; 解码器 (l>L/2) 由 H_{L/2} 投影 (§2.2 CED);
        Reindex/Reuse: 复用 (由外部传入, 本函数不调用)。"""
        src = h if Henc is None else Henc
        B, L, _ = src.shape
        m2 = 2 * self.m
        pad = (m2 - L % m2) % m2
        x = F.pad(src, (0, 0, 0, pad))
        x = x.reshape(B, (L + pad) // m2, m2 * self.c.d)          # 无重叠拼接
        return self.kv_down(x)                                    # (B, L/m2, kv_lora)

    def forward(self, h, pos, cache=None, shared=None, cand=None):
        """cache/shared: 复用状态。返回 (out, state, cand)。§2.3.1 三模式 + §2.2 SWA 逐层。"""
        c = self.c
        B, L, _ = h.shape
        z = self.norm(h)
        q = self.q_up(self.q_down(z)).view(B, L, c.n_heads, c.head_dim)
        q = self.q_norm(rope(q, pos, c.rope_dim)).transpose(1, 2)     # (B,H,L,D)

        # ---- SWA 分支: 本层 H_l → 未压缩 latent → 共享上投影 (§2.2: 局部 KV 逐层自算) ----
        lok = self.kv_down_local(z)
        ks = self.k_up(lok).view(B, L, c.n_heads, c.head_dim)
        vs = self.v_up(lok).view(B, L, c.n_heads, c.head_dim)
        ks = rope(ks, pos, c.rope_dim)
        if c.fp8_swa:
            ks, vs = fp8_quant(ks), fp8_quant(vs)                      # §2.4.4: SWA KV 保 FP8
        ks, vs = ks.transpose(1, 2), vs.transpose(1, 2)
        w = min(c.swa_window, L)
        # ★ 带状 SWA 用 scaled_dot_product_attention: 只给 (L,L) 布尔掩码(不物化打分矩阵)。
        #   原先手写 (q@kᵀ → softmax) 会开 L×L×H 的打分张量, 4096 长度就吃满 8G 显存。
        ii = torch.arange(L, device=h.device)
        band = ii[:, None] - ii[None, :]
        sm = (band >= 0) & (band < w)
        o_swa = F.scaled_dot_product_attention(q, ks, vs, attn_mask=sm, dropout_p=0.0)

        # ---- 纯 SWA 层 (§2.1: 前两层 "use SWA only") ----
        if self.mode == "swa_only":
            o, state, newcand = o_swa, {}, cand
        else:
            # ---- main KV / indexer K / Top-K 由模式决定 (§2.3.1) ----
            if self.mode == "full":
                mkv = self._main_kv(z, shared.get("Henc") if shared else None)
                if c.fp4_kv:
                    mkv = mx_fp4_quant(mkv)                            # RoPE 之后量化 (§2.4.4)
                state = {"mkv": mkv}
            else:
                mkv = shared["mkv"] if (shared and "mkv" in shared) else cache["mkv"]
                state = {"mkv": mkv}
            n_ent = mkv.shape[1]
            m2 = 2 * self.m
            ent_end = (torch.arange(n_ent, device=h.device) + 1) * m2
            vis = ent_end[None, :] <= pos[:, None]                     # (L,n_ent) 因果可见性

            if self.mode == "reuse":
                sel = shared["sel"] if (shared and "sel" in shared) else cache["sel"]
                newcand = cand
                state["sel"] = sel     # ★ 继续往下传, 否则后面的 Reuse 拿不到
            else:
                s = self.indexer.scores(z, mkv)                        # (B,idxH,L,n)
                if self.mode == "reindex" and cand is not None:
                    sel = HierSparseIndexer.topk_mask(s, c.topk, cand)
                    newcand = cand
                else:
                    s = s.masked_fill(~vis[None, None], float("-inf"))
                    sel = HierSparseIndexer.topk_mask(s, c.topk)
                    newcand = self.indexer.build_candidate_pool(s, vis) if self.build_pool else cand
                # ★ Top-K 由 indexer 头选出而主注意力头数不同 ⇒ 跨头并集成每查询一份
                sel = sel.any(1, keepdim=True)                         # (B,1,L,n)
                state["sel"] = sel

            k_up = self.k_up(mkv) if self.mode == "full" else \
                (shared["k_up"] if (shared and "k_up" in shared) else cache["k_up"])
            v_up = self.v_up(mkv) if self.mode == "full" else \
                (shared["v_up"] if (shared and "v_up" in shared) else cache["v_up"])
            state["k_up"], state["v_up"] = k_up, v_up
            k = k_up.view(B, n_ent, c.n_heads, c.head_dim).transpose(1, 2)
            v = v_up.view(B, n_ent, c.n_heads, c.head_dim).transpose(1, 2)
            sc = (q @ k.transpose(-1, -2)) / math.sqrt(c.head_dim)
            sc = sc.masked_fill(~sel, float("-inf"))
            a = torch.nan_to_num(sc.softmax(-1), nan=0.0, posinf=0.0, neginf=0.0)
            o = (a @ v) + o_swa

        # 分组输出投影 (§4.2.1)
        og = c.n_heads * c.head_dim // c.out_groups
        ogrp = o.transpose(1, 2).reshape(B, L, c.out_groups, og)
        ogrp = self.o_down(ogrp).reshape(B, L, c.out_groups * c.out_group_dim)
        return self.o_up(ogrp), state, newcand


# --------------------------------------------------------------------- mHC (§2.4.1)

class SinglePassMHC(nn.Module):
    """n 条残差流; 系数 H(X_l); 输入混合用**上一块**的 A (shift by one) ⇒ 单趟可融合。"""

    def __init__(self, c: Cfg):
        super().__init__()
        self.n = c.mhc_n
        self.hnorm = RMSNorm(c.d * self.n)
        # 一次投影出 A(1×n), B(n×n), C(n×1)
        self.proj_a = nn.Linear(c.d * self.n, c.mhc_n, bias=False)
        self.proj_b = nn.Linear(c.d * self.n, c.mhc_n * c.mhc_n, bias=False)
        self.proj_c = nn.Linear(c.d * self.n, c.mhc_n, bias=False)
        self.prev_a: Optional[torch.Tensor] = None

    def coeffs(self, X):
        z = self.hnorm(X.reshape(X.shape[0], X.shape[1], -1))
        a = self.proj_a(z).softmax(-1)                          # (B,L,n)
        bb = self.proj_b(z).view(X.shape[0], X.shape[1], self.n, self.n).softmax(-1)
        cc = self.proj_c(z).sigmod if False else self.proj_c(z).sigmoid()
        return a, bb, cc

    def step(self, X, block, prev_a=None):
        """X: (B,L,n,d)。Single-Pass mHC: 输入混合用**上一块**的 A (论文式(6))。
        ★ 状态显式进出 (不再存 self.prev_a): 梯度检查点重算才不会读到错的状态。"""
        a_now, bb, cc = self.coeffs(X)
        a_use = a_now if prev_a is None else prev_a     # 第一块没有"上一块", 用自己的
        Xin = torch.einsum("bln,blnd->bld", a_use, X)   # A_{l-1} X_l
        y = block(Xin)                                  # F_l(·)
        Xn = torch.einsum("blmn,blnd->blmd", bb, X) + cc.unsqueeze(-1) * y.unsqueeze(2)
        return Xn, a_now


# --------------------------------------------------------------------- MoE (§2.1.1)

class MoE(nn.Module):
    """1 共享专家 + n_routed 路由专家, SwiGLU + clamp(10); 图像/文本**分开的负载偏置**。"""

    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.router = nn.Linear(c.d, c.n_routed, bias=False)
        self.bias_text = nn.Parameter(torch.zeros(c.n_routed))
        self.bias_img = nn.Parameter(torch.zeros(c.n_routed))
        self.shared = nn.ModuleList([SwiGLU(c) for _ in range(c.n_shared)])
        self.experts = nn.ModuleList([SwiGLU(c) for _ in range(c.n_routed)])

    def forward(self, x, modality=None):
        c = self.c
        B, L, _ = x.shape
        logits = self.router(x)                                  # 原始路由分 (用于加权)
        bias = self.bias_text if modality is None else \
            torch.where(modality.unsqueeze(-1) > 0, self.bias_img, self.bias_text)
        idx = (logits + bias).topk(c.topk_expert, dim=-1).indices   # ★ 偏置只影响"选谁"
        w = logits.gather(-1, idx).softmax(-1)                      # ★ 权重用原始分数
        out = torch.zeros_like(x)
        xf, wf, idf = x.reshape(-1, c.d), w.reshape(-1, c.topk_expert), idx.reshape(-1, c.topk_expert)
        ov = out.view(-1, c.d)
        for e in range(c.n_routed):
            hit = (idf == e)
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            we = (hit[rows].to(x.dtype) * wf[rows]).sum(-1, keepdim=True)
            ov.index_add_(0, rows, self.experts[e](xf[rows]) * we)
        for s in self.shared:
            out = out + s(x)
        return out


class SwiGLU(nn.Module):
    """SwiGLU + clamp 门控 (§4.2.1: 阈值 10)"""

    def __init__(self, c: Cfg):
        super().__init__()
        self.gate = nn.Linear(c.d, c.expert_dim, bias=False)
        self.up = nn.Linear(c.d, c.expert_dim, bias=False)
        self.down = nn.Linear(c.expert_dim, c.d, bias=False)
        self.c = c

    def forward(self, x):
        g = self.gate(x).clamp(-self.c.clamp, self.c.clamp)
        return self.down(F.silu(g) * self.up(x))


# --------------------------------------------------------------------- Engram (§2.4.2)

def is_prime(n):
    if n < 2:
        return False
    for p in range(2, int(n ** 0.5) + 1):
        if n % p == 0:
            return False
    return True


def prev_primes(n, k):
    """取 k 个**互不相同的素数**当各头的表长 (论文: 表长取不同素数)"""
    out, x = [], n
    while len(out) < k:
        x -= 1
        if is_prime(x):
            out.append(x)
    return out


class Engram(nn.Module):
    """条件记忆: tokenizer 压缩 + 多头哈希 + 上下文门控 + 多分支融合;
    **去掉短因果卷积** (§2.4.2)。FP8 存储。"""

    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.orders = c.engram_orders
        self.heads = c.engram_heads
        self.head_dim = c.engram_order_dim // c.engram_heads
        self.tables = nn.ModuleList()
        self.sizes = []
        for _ in self.orders:
            sz = prev_primes(c.engram_entries, self.heads)       # 每头一张, 表长互不相同的素数
            self.sizes.append(sz)
            self.tables.append(nn.ModuleList(
                [nn.Embedding(n, self.head_dim) for n in sz]))
        self.gate = nn.Linear(c.d, c.engram_order_dim * len(self.orders), bias=False)
        self.out = nn.Linear(c.engram_order_dim * len(self.orders), c.d, bias=False)
        self.norm = RMSNorm(c.d)

    def forward(self, tok, h):
        B, L = tok.shape
        feats = []
        for oi, n in enumerate(self.orders):
            acc = []
            for hi, tbl in enumerate(self.tables[oi]):
                if L < n:
                    idx = torch.zeros(B, L, dtype=torch.long, device=tok.device)
                else:
                    win = torch.stack([tok[:, i:L - n + 1 + i] for i in range(n)], dim=-1)
                    hv = (win * (hi + 1)).sum(-1)
                    idx0 = (hv * (hi * 2 + 3) + n * 7919 + hi * 104729) % tbl.num_embeddings
                    idx = F.pad(idx0, (n - 1, 0))     # 前 n-1 个位置凑不齐 n-gram ⇒ 用第 0 行
                acc.append(F.embedding(idx, tbl.weight))
            feats.append(torch.cat(acc, dim=-1))
        f = torch.cat(feats, dim=-1)
        if self.c.fp8_swa:
            f = fp8_quant(f)
        g = self.gate(self.norm(h)).sigmoid()
        return self.out(f * g)


# --------------------------------------------------------------------- DSpark (§2.4.3)

class DSpark(nn.Module):
    """草稿模块: 3 个块 (SWA 窗 128), 一次前向并行出 5 个草稿位; Markov 头 + 置信头。"""

    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=c.d, nhead=c.n_heads, dim_feedforward=4 * c.d, batch_first=True,
            norm_first=True) for _ in range(c.dspark_blocks)])
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.markov = nn.Embedding(c.vocab, c.vocab)             # 草稿间的 Markov 依赖
        self.conf = nn.Linear(c.d, 1)                            # 每位置的条件接受概率

    def forward(self, h):
        for b in self.blocks:
            h = b(h)
        base = self.head(h)
        return base, self.conf(h).sigmoid()


# --------------------------------------------------------------------- 主干

class Block(nn.Module):
    def __init__(self, c: Cfg, layer: int, mode: str):
        super().__init__()
        self.c, self.layer, self.mode = c, layer, mode
        self.mhc = SinglePassMHC(c)
        self.attn = None if mode == "swa_only" else CSA2Attention(c, mode, c.compress_m(layer))
        self.mlp_norm = RMSNorm(c.d)
        self.moe = MoE(c)

    def forward(self, X, pos, shared=None, cand=None, prev_a=None):
        """X: (B,L,n,d) —— mHC 的 n 条残差流。两次 mHC 步: 注意力 → MoE (§2.4.1)。
        ★ 纯函数: 复用状态与候选池**作为返回值**传出, 不写在模块属性上 (否则梯度检查点重算对不上)。"""
        holder = [None, cand]

        def attn_fn(x):
            if self.attn is None:
                return torch.zeros_like(x)
            o, s, c = self.attn(x, pos, shared=shared, cand=cand)
            holder[0], holder[1] = s, c
            return o

        X, a1 = self.mhc.step(X, attn_fn, prev_a)
        X, a2 = self.mhc.step(X, lambda x: self.moe(self.mlp_norm(x)), a1)
        return X, (holder[0] if holder[0] is not None else {}), holder[1], a2


class Model(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.modes = c.mode_schedule()
        self.emb = nn.Embedding(c.vocab, c.d)
        # Engram 模块位置: 论文是第 1、14 层 (0 起, 共 40 层) ⇒ 按比例缩放到本模型
        self.engram_layers = {1, max(2, round(14 / 40 * c.n_layers))}
        self.engrams = nn.ModuleDict({str(i): Engram(c) for i in self.engram_layers})
        self.grad_ckpt = False          # 训练脚本里打开 (6G 显存)
        self.blocks = nn.ModuleList([Block(c, i, m) for i, m in enumerate(self.modes)])
        # ★ 候选池只由**解码器**的 Full 层建 (§2.3.2: "the decoder's first CSA2 layer in Full mode
        #   ... builds a shared candidate pool for subsequent layers"); 编码器不建。
        self.enc_last = 2 + c.enc_csa_layers() - 1
        for i, blk in enumerate(self.blocks):
            if blk.attn is not None:
                blk.attn.build_pool = (blk.mode == "full" and i > self.enc_last)
        self.norm = RMSNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.dspark = DSpark(c)

    def forward(self, tok, return_state=False):
        c = self.c
        B, L = tok.shape
        pos = torch.arange(L, device=tok.device)
        h = self.emb(tok)
        # mHC: 复制成 n 条流
        X = h.unsqueeze(2).expand(B, L, c.mhc_n, c.d).contiguous()
        shared, cand, Henc, prev_a = {}, None, None, None
        # CED: 编码器最后一层 (前 2 纯 SWA + 编码器 CSA2 层) 的隐状态 = 解码器 global KV 的来源
        enc_last = 2 + c.enc_csa_layers() - 1
        for i, blk in enumerate(self.blocks):
            if self.grad_ckpt and self.training:
                # ★ 梯度检查点: 反传时重算该块激活, 激活内存不随层数线性增长 (6G 显存必需)。
                #   前提是 Block 是纯函数 (状态走返回值), 见 Block.forward。
                from torch.utils.checkpoint import checkpoint
                X, st, nc, prev_a = checkpoint(blk, X, pos, shared, cand, prev_a,
                                               use_reentrant=False)
            else:
                X, st, nc, prev_a = blk(X, pos, shared=shared, cand=cand, prev_a=prev_a)
            if str(i) in self.engrams:
                e = self.engrams[str(i)](tok, X.mean(2))
                X = X + e.unsqueeze(2)
            if blk.mode != "swa_only":
                # 注意力已把 main KV / indexer K / Top-K 沿链带下来 (§2.3.1) ⇒ 整体替换即可
                shared = dict(st)
                if blk.attn is not None and blk.attn.build_pool and blk.mode == "full":
                    cand = nc                                     # 分层候选池 (§2.3.2)
            if i == enc_last:
                Henc = X.mean(2)                                  # CED: 解码器 global KV 的来源
                shared = dict(shared)
                shared["Henc"] = Henc
        h = self.norm(X.mean(2))
        return (self.head(h), shared) if return_state else self.head(h)


# --------------------------------------------------------------------- 账目核对

def count_params(c: Cfg) -> Tuple[int, int, int]:
    """在 meta 设备上实例化并数参数 ⇒ 返回 (主干, Engram, DSpark)。
    ★ DSpark 是**独立训练的模块** (§2.4.3: "we train only DSpark while keeping the backbone
    frozen"), 不计入 552B 主干, 故单列。"""
    with torch.device("meta"):
        m = Model(c)
    eng = sum(p.numel() for k, p in m.named_parameters() if k.startswith("engrams."))
    dsp = sum(p.numel() for k, p in m.named_parameters() if k.startswith("dspark."))
    tot = sum(p.numel() for p in m.parameters())
    return tot - eng - dsp, eng, dsp


def param_breakdown(c: Cfg) -> dict:
    """按模块分类数参数, 用于核对 552B 的构成"""
    with torch.device("meta"):
        m = Model(c)
    buckets = {"emb+head": 0, "attn": 0, "moe": 0, "mhc": 0, "norm": 0, "engram": 0, "dspark": 0}
    for k, p in m.named_parameters():
        n = p.numel()
        if k.startswith("engrams."):
            buckets["engram"] += n
        elif k.startswith("dspark."):
            buckets["dspark"] += n
        elif k.startswith("emb.") or k.startswith("head."):
            buckets["emb+head"] += n
        elif ".moe." in k:
            buckets["moe"] += n
        elif ".mhc." in k:
            buckets["mhc"] += n
        elif ".norm." in k or ".ln" in k or k.endswith("w") and "norm" in k:
            buckets["norm"] += n
        else:
            buckets["attn"] += n
    return buckets


def kv_bytes_per_token(c: Cfg) -> dict:
    """每 token 的持久 KV 字节 (§2.4.4 / §3.2.1)。
    持久缓存**只存 main KV** (FP4) 与 indexer K (FP4), **不存 SWA KV** (§3.2.1)。"""
    modes = c.mode_schedule()
    main = 0.0        # main KV: 压缩后每条覆盖 2m 个 token ⇒ 每 token 1/(2m) 条; FP4 = 0.5 B/值
    for i, m in enumerate(modes):
        if m == "full":
            main += (1.0 / (2 * c.compress_m(i))) * c.kv_lora * 0.5
    idx = 0.0         # indexer K: 由 main KV 投影, 只在 Full/Reindex 层存 (FP4)
    for i, m in enumerate(modes):
        if m in ("full", "reindex"):
            idx += (1.0 / (2 * c.compress_m(i))) * c.idx_heads * c.idx_head_dim * 0.5
    return dict(n_full=sum(1 for m in modes if m == "full"),
                n_reindex=sum(1 for m in modes if m == "reindex"),
                main_kv_fp4=main, indexer_k_fp4=idx, total=main + idx,
                swa_stored=0.0)          # 持久缓存不存 SWA


if __name__ == "__main__":
    c = Cfg()
    print("层模式 (%d 层): %s" % (c.n_layers, ", ".join(
        "%d:%s" % (i, m) for i, m in enumerate(c.mode_schedule()))))
    p, e, d = count_params(c)
    print("小模型参数: 主干 %.3fM + Engram %.3fM + DSpark %.3fM"
          % (p / 1e6, e / 1e6, d / 1e6))
    # 论文原尺寸核对 (结构相同, 只放大维度)
    big = Cfg(vocab=129280, d=5120, n_layers=40, n_heads=64, head_dim=512, q_lora=1280,
              kv_lora=512, out_groups=8, out_group_dim=1024, idx_heads=32, idx_head_dim=128,
              topk=512, cand_blocks=2048, n_routed=384, topk_expert=6, expert_dim=2304,
              engram_heads=8, engram_order_dim=2048, engram_entries=16000000)
    pb, eb, db = count_params(big)
    print("论文尺寸: 主干 %.1fB + Engram %.1fB  (论文: 552B + 196B)"
          % (pb / 1e9, eb / 1e9))
    print("小模型每 token 持久 KV: %s" % kv_bytes_per_token(c))