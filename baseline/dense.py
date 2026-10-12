"""稠密基线: 标准 Transformer LM (同 d/层数/头数, RoPE + 因果注意力), 用于与 Engram-Rule 做 A/B。
参数预算与 engram_rule 的模型对齐 (同 RCfg), 唯一差别 = 没有 Engram/规则路径。"""
import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engram_rule"))
from model import RCfg, rope


class DenseBlock(nn.Module):
    def __init__(self, c: RCfg):
        super().__init__()
        self.c = c
        self.n1 = nn.LayerNorm(c.d)
        self.n2 = nn.LayerNorm(c.d)
        self.q = nn.Linear(c.d, c.n_heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.d, c.n_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.d, c.n_heads * c.head_dim, bias=False)
        self.o = nn.Linear(c.n_heads * c.head_dim, c.d, bias=False)
        self.ffn = nn.Sequential(nn.Linear(c.d, c.ffn_mult * c.d), nn.GELU(),
                                 nn.Linear(c.ffn_mult * c.d, c.d))

    def forward(self, h, pos):
        c = self.c
        B, L, _ = h.shape
        z = self.n1(h)
        q = rope(self.q(z).view(B, L, c.n_heads, c.head_dim), pos, c.rope_dim).transpose(1, 2)
        k = rope(self.k(z).view(B, L, c.n_heads, c.head_dim), pos, c.rope_dim).transpose(1, 2)
        v = self.v(z).view(B, L, c.n_heads, c.head_dim).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.o(att.transpose(1, 2).reshape(B, L, c.n_heads * c.head_dim))
        return h + self.ffn(self.n2(h))


class DenseLM(nn.Module):
    def __init__(self, c: RCfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList([DenseBlock(c) for _ in range(c.n_layers)])
        self.norm = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h, pos)
        return self.head(self.norm(h))

    def stats(self):
        return dict(rule_skip_ratio=float("nan"), alive_slots=0, n_slots=0)

    def export_rules(self, topk=20):
        return []
