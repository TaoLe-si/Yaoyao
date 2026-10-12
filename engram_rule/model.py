"""Engram-Rule 神经符号 Transformer (按 v1.0 指导书实现)。

裁定见 engram_rule/DESIGN.md (指导书里二义的地方逐条定死), 摘要:
  · 多哈希只取候选槽 (O(1)), 可微软读**只在候选集内**做 ⇒ 同时满足 O(1) 与可微
  · 哈希键取 **token id** (§4.2 公式); n-gram 向量(软读询问)取可训练嵌入
  · f(v_k,i,j) = 规则向量 → 相对位置分桶偏置 bucket_proj(v_k)[bucket(i-j)]
  · 门控 g_k = σ(w_k ⊙ v_k) (w_k 为 d 维), 另有标量控制强度
  · 规则路径与内容路径分开计数; 该位置**所有哈希都命中空槽**时规则路径整体跳过
  · 空槽由**语料 n-gram 统计**决定 (指导书 §5.3 阶段 1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- 配置

@dataclass
class RCfg:
    vocab: int = 8192
    d: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_mult: int = 4
    rope_dim: int = 32
    # --- Engram ---
    ngram_max: int = 3          # n-gram 阶 1..3
    n_hash: int = 3             # 多哈希个数
    n_slots: int = 65536        # Engram 槽位数 N
    beta: float = 1.0           # 软读温度 (softmax(-β‖k-k_i‖²))
    min_count: int = 2          # 语料统计: 出现 ≥ 该次数的 n-gram 才占槽 (§5.3 阶段1)
    # --- 规则 ---
    n_buckets: int = 64         # 相对位置分桶数 (★ 越大窗口越窄: W = k_buckets*L/nb)
    k_buckets: int = 4          # 每条查询选几个桶 (选"看哪里")
    use_pos_abs: bool = True    # 开关: 关掉后门的查询侧只用 RoPE 的相对位置 (供 A/B 判它到底有没有用)
    pos_abs_max: int = 512      # ★ 显式绝对位置表的长度 (给门的查询侧一条位置通路)
                                #   理由(排除法): 双峰的成因是"必须知道 k"(前缀里有几个 PP),
                                #   而 k 由查询的绝对位置决定(vpos=2+3k); RoPE 是**相对**的、
                                #   对平移不变, "从旋转相位解出绝对位置"正是它不擅长的事 ⇒ 学不稳。
                                #   代价是一张 max_L×Dg 的小表, 不破坏线性代价与固定窗口。
    use_pos_bucket: bool = True # ★ 路线①(b): 位置(可数统计) → **桶偏置**(窗口起点) 的通路开关。
                                #   §25 判决: 起点只由查询的**局部 n-gram** 决定, 而"中心词有多远"
                                #   (=1+3k) 不在局部 n-gram 里 ⇒ 规则没有信息把窗口放对位置。
                                #   pos_abs 只进门的**查询侧**(窗内怎么挑), 到不了"看多远"这一环。
    clamp_window: bool = True   # ★ 因果不变量 (C28): 起点偏移夹到 [max(0,pos-W+1), pos], 避免空窗。
                                #   旧行为可使窗口整段越过因果前缀 ⇒ 窗口聚合支路无有效 K/V、该支路梯度为 0。
                                #   §4.4 direct(agg) 是独立旁路，不能把它与窗口聚合支路混为一谈。
    near_first: float = 2.0     # ★ 归纳偏置: 桶偏置初值偏向**近端** (bias[b] ∝ -b)。
                                #   理由(静态分析): 桶选择只由查询局部 n-gram 决定, 而"中心词距离"
                                #   随 k 变(1/4/7)且不在局部 n-gram 里 ⇒ 规则只能固定选一组桶;
                                #   初值随机 ⇒ 约半数种子的窗口盖不住中心词 ⇒ 学成词项关联(对未见词全错)
                                #   ⇒ 实测正确率极差 0.978 的双峰。偏向近端让初始窗口必含 {1,4,7}。
    # --- ★问题池化通路 (§61) ---
    #   触发: §61.2 实测答案第 1 步的窗口 [win_lo, win_lo+16) 覆盖问题末尾的比例是
    #   **0/20**, 且 start_bucket 只取 {0,1} ⇒ 问题在结构上不可达。
    #   §61.3 C 读数: 把问题**乱序**后生成改变的行只有 3/20 ⇒ 模型确实没在读问题内容。
    #   修法(预登记于 §61.5): 把问题段池化成一个向量, 广播加到每个位置 ——
    #   O(L) 成本, **不改**窗口/Engram/pos_bucket ⇒ 与 §53 C3 的 O(1) 主张、
    #   §47 的"窗口不是瓶颈"都不冲突。
    use_qpool: bool = False     # ★ 默认 **关闭**: 不出手改动出货路径(规则 3)
    qpool_alpha_init: float = 0.1   # 可学习标量初值(小值 ⇒ 模型自己决定用多少)
    qpool_mode: str = "mean"    # mean | max —— 池化方式
    # ★问题段的 token id。**不硬编码**: 从 checkpoint 的 tokenizer 查出来存进 cfg。
    #   -1 = 没给 ⇒ 通路自动退化为"手动传 q_pool"模式(行为与旧版一致)。
    qpool_user_id: int = -1     # [USER] 的 id
    qpool_assist_id: int = -1   # [ASSISTANT] 的 id
    gate_rank: int = 16         # ★ 路线1: 门控低秩维 Dg。代价 ∝ D·Dg ⇒ Dg=16 比 Dg=64 便宜 4 倍
    local_norm: bool = True     # ★ 路线3: 窗口内局部归一化 (O(W), 把 softmax 的"凸组合"性质拿回来)
    window_max: int = 16        # ★ 固定窗口上限: 所有训练/推理调用使用同一 W。
    bucket_step: int = 0        # 每桶对应的回看距离步长; 0=legacy 动态步长, 新 checkpoint 必须显式保存非零值。
    rule_scale_init: float = 2.0
    lam_init: float = 1.0

    def __post_init__(self):
        # ★ 配置合法性: 静态分析时漏掉的一条 —— rope_dim 超过 head_dim 会让 x[..., dim:] 变空,
        #   与前半段 cat 时报 "size of tensor a (0) must match b (16)"。必须前置断言。
        assert self.d % self.n_heads == 0, "d 必须能被 n_heads 整除"
        assert 0 < self.rope_dim <= self.head_dim,             "rope_dim(%d) 必须 <= head_dim(%d)" % (self.rope_dim, self.head_dim)
        assert self.rope_dim % 2 == 0, "rope_dim 需为偶数"
        assert self.bucket_step >= 0, "bucket_step 必须为0(legacy)或正整数"

    @property
    def head_dim(self) -> int:
        return self.d // self.n_heads

    @property
    def n_cand(self) -> int:
        """每位置检索的候选规则数 R = 阶数 × 哈希数"""
        return self.ngram_max * self.n_hash


# --------------------------------------------------------------------- 工具

_ROPE_CACHE = {}
_ARANGE_CACHE = {}
_ARANGE_CACHE_CAP = 4096        # ★ 规则 9: 缓存必须有硬上限


def _arange(n, device):
    """★ §51: 窗口索引用的常量张量缓存。torch.arange(n) 的值与 n 一一对应,
    缓存后逐位相同(纯整数, 不涉及浮点归约), 但省掉一次 kernel 提交。"""
    key = (int(n), str(device))
    t = _ARANGE_CACHE.get(key)
    if t is None:
        if len(_ARANGE_CACHE) >= _ARANGE_CACHE_CAP:
            _ARANGE_CACHE.clear()
        t = torch.arange(n, device=device)
        _ARANGE_CACHE[key] = t
    return t


def _rope_tables(L, dim, device, dtype, theta=10000.0):
    """★ 性能: cos/sin 表按 (L, dim, 设备, dtype) 缓存成 buffer。
    原实现每次调用现算 arange/幂/cos/sin, 而前向要调 8 次 (4 层 × q,k) ⇒ 每次省 5+ 个算子。"""
    key = (L, dim, str(device), dtype)
    t = _ROPE_CACHE.get(key)
    if t is None:
        half = dim // 2
        f = torch.arange(half, device=device, dtype=torch.float32) / half
        inv = theta ** (-f)
        ang = torch.arange(L, device=device, dtype=torch.float32).view(L, 1) * inv.view(1, half)
        c = ang.cos().to(dtype).view(1, L, 1, half)
        s = ang.sin().to(dtype).view(1, L, 1, half)
        t = (torch.cat([c, c], -1), torch.cat([s, s], -1))   # 前后半段公用 ⇒ 一次 mul 打完
        _ROPE_CACHE[key] = t
    return t


def rope(x, pos, dim, theta=10000.0):
    """部分 RoPE, x: (B,L,H,D), 只转前 dim 个通道。
    ★ 性能: 用 rotate_half 形式 + 预计算表: mul + mul + add 三步, 不再逐次算 cos/sin 与多次切片。"""
    B, L, H, D = x.shape
    assert pos is not None and int(pos[0]) == 0 and int(pos[-1]) == L - 1,         "预计算表假设 pos = arange(L); 其他位置编码需回退到逐次计算"
    cfull, sfull = _rope_tables(L, dim, x.device, x.dtype, theta)
    half = dim // 2
    xr, xp = x[..., :half], x[..., half:dim]
    rot = torch.cat([-xp, xr], dim=-1)                       # rotate_half
    y = x[..., :dim] * cfull + rot * sfull
    return torch.cat([y, x[..., dim:]], dim=-1) if dim < D else y


_HASH_MULT = (2654435761, 40503, 2246822519)
_HASH_MOD = (4294967291, 4294967279, 4294967231)


class EngramMemory(nn.Module):
    """可微键值记忆: 槽位 → (键向量, 值向量) + 存活位。
    多哈希给出候选槽 (O(1)), 软读只在候选集内 (可微)。"""

    def __init__(self, c: RCfg):
        super().__init__()
        self.c = c
        self.key = nn.Embedding(c.n_slots, c.d)
        self.val = nn.Embedding(c.n_slots, c.d)
        self.register_buffer("alive", torch.zeros(c.n_slots, dtype=torch.bool))
        # ★ 性能修复: 乘子/模数**预计算成 buffer**。原实现每次前向在 Python 双重循环里
        #   torch.tensor([pow(...)]) 新建并搬上设备 ⇒ 9 次 host 分配 + 9 次 H2D 拷贝/前向。
        nm = c.ngram_max
        mm = torch.zeros(nm, c.n_hash, nm, dtype=torch.int64)
        md = torch.ones(nm, c.n_hash, 1, dtype=torch.int64)
        for oi, n in enumerate(range(1, nm + 1)):
            for hi in range(c.n_hash):
                m_, P_ = _HASH_MULT[hi % len(_HASH_MULT)], _HASH_MOD[hi % len(_HASH_MOD)]
                for k in range(n):                      # 第 k 个(自最老起)的权重 = m^(k+1)
                    mm[oi, hi, nm - n + k] = pow(m_, k + 1, P_)
                md[oi, hi, 0] = P_
        self.register_buffer("h_mult", mm)
        self.register_buffer("h_mod", md)
        self._any_alive_cpu = False                      # 缓存"有无存活槽", 免去每步设备同步
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.val.weight, std=0.02)

    # ---------- 多哈希 ----------
    def slot_ids(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,L) token id → 候选槽 (B,L,R), R = 阶数×哈希数。同 n-gram 必得同槽。
        ★ 性能修复: 原版是 Python 双重循环 (3 阶 × 3 哈希), 每轮 pad+unfold+建张量+乘+模
          ⇒ 每次前向约 54 个 kernel + 9 次 H2D。现用**一次 einsum** 把所有阶/哈希算完
          (权重已按列对齐预存在 h_mult, 短阶的高位列权重为 0 ⇒ 与旧实现**逐位等价**, 见 C16)。"""
        c = self.c
        B, L = x.shape
        nm = c.ngram_max
        win = F.pad(x, (nm - 1, 0)).unfold(1, nm, 1)          # (B,L,nm) 以 t 结尾的 nm 窗口
        # ★ C17 抓出: int64 的 einsum 在 CUDA 上是 "baddbmm_cuda not implemented for Long"
        #   ⇒ 我那个"性能修复"在 GPU 上会直接崩。改成**广播乘 + 求和**(逐元素算子支持 int64,
        #   且张量只有 B·L·nm·阶·哈希 个元素, 极小)。
        # ★ 轴配对: 要的是 hv[b,l,o,h] = Σ_p win[b,l,p]·mm[o,h,p]
        #   ⇒ win 弄成 (B,L,1,1,nm) [p 在最后], mm 弄成 (1,1,no,nh,nm), 对最后一维求和。
        #   (第 1 版写成 win.unsqueeze(-1).unsqueeze(-1) * h_mult ⇒ 把 mm 的**阶轴**对上了
        #    win 的**位置轴**, 哈希全错、碰撞率 0.87 —— C6b 抓出。)
        no, nh = self.h_mult.shape[0], self.h_mult.shape[1]
        hv = (win.to(torch.int64).view(B, L, 1, 1, nm)
              * self.h_mult.view(1, 1, no, nh, nm)).sum(-1)          # (B,L,阶,哈希)
        hv = hv % self.h_mod.view(1, 1, nm, c.n_hash)
        return (hv.reshape(B, L, -1) % c.n_slots).to(torch.int64)

    def ngram_query(self, x: torch.Tensor, emb: nn.Embedding,
                    b_idx=None, t_idx=None) -> torch.Tensor:
        """软读的询问向量: n-gram 内 token 嵌入之和 (可训练)。
        ★ 性能: 原实现对 nm 个阶各做一次嵌入查表+求和 (3 次查表)。现改成
          **一次查表(整个 nm 窗口) + 一次后缀和** —— 阶 n 的询问 = 窗口最后 n 个 token 的嵌入和。"""
        c = self.c
        B, L = x.shape
        if b_idx is None:
            b_idx = torch.arange(B, device=x.device).repeat_interleave(L)
            t_idx = torch.arange(L, device=x.device).repeat(B)
        nm = c.ngram_max
        toks = torch.stack([x[b_idx, (t_idx - j).clamp(min=0)] for j in range(nm - 1, -1, -1)],
                           dim=-1)                                    # (M,nm) 窗口 [t-nm+1..t]
        es = emb(toks)                                                # (M,nm,d)
        suf = es.flip(1).cumsum(1).flip(1)                            # 后缀和: suf[:,k]=Σ_{j>=k}
        qs = []
        for n in range(1, nm + 1):
            qs.append(suf[:, nm - n])                                 # 阶 n = 最后 n 个之和
        q = torch.stack(qs, dim=1)                                    # (M,阶,d)
        return q.repeat_interleave(c.n_hash, dim=1)                   # (M,R,d) 每阶复制 n_hash 份

    # ---------- 语料统计初始化 (§5.3 阶段 1) ----------
    @torch.no_grad()
    def build_from_corpus(self, tokens: torch.Tensor, batch: int = 64) -> Dict[str, float]:
        """扫语料, 把出现次数 ≥ min_count 的 n-gram 所占槽位标为存活 ⇒ 空槽是真实的。"""
        # ★ 设备对齐: 乘子改为模块 buffer 后, 传 CPU 语料给 CUDA 模块会 einsum 设备不匹配
        #   (旧实现乘子是 device=x.device 现建, 所以自动匹配)。这属于**静态可预见**的错误,
        #   我却是跑基准时才撞到 —— 违反"禁止错误修正前跑代码"。统一在此对齐设备。
        tokens = tokens.to(self.alive.device)
        cnt = torch.zeros(self.c.n_slots, dtype=torch.long, device=self.alive.device)
        L = tokens.shape[1]
        for i in range(0, L, batch):
            chunk = tokens[:, i:i + batch]
            if chunk.shape[1] < 2:
                continue
            ids = self.slot_ids(chunk).reshape(-1)
            cnt.index_add_(0, ids, torch.ones_like(ids, dtype=torch.long))
        self.alive.copy_(cnt >= self.c.min_count)
        return dict(alive_slots=int(self.alive.sum()),
                    n_slots=self.c.n_slots,
                    alive_ratio=float(self.alive.float().mean()))

    # ---------- 可微读 ----------
    def read(self, x: torch.Tensor, emb: nn.Embedding):
        """返回 (逐候选规则向量 (B,L,R,d), 软读权重 (B,L,R), 是否跳过 (B,L))。
        ★ 不在这里求和: §4.4 要求**先对每条规则门控、再加权求和**, 顺序不能反
          (先把候选聚成一条再门控会破坏"冲突加权=加权和"这条可读性)。

        ★ §51 性能: 原实现用 `slot.reshape(-1,R)[flat]` / `v_out...[flat] = v` 这类
          **布尔掩码索引**只取存活位。CUDA 上布尔掩码索引要走 nonzero ⇒ 每处一次
          device→host 同步。实测单次前向 15 次 aten::nonzero / 26 次
          cudaStreamSynchronize, 每次同步 49.5µs ⇒ 光同步就吃掉前向 host 时间的三分之二。
          现改成**全张量算 + any_live 掩码**: 逐位等价(判据 check_fast_path_equiv.py
          的 E1, 含"漏掉 any_live 掩码"的负向对照), 但一次同步都不产生。

        ★ `if bool(any_live.any())` 这道闸**故意保留**: 它是 C10 的判据(全空槽时连
          n-gram 嵌入都不算); 且不能缓存 alive 标志 —— 直接改 alive(检查脚本就改)
          会让缓存失效。"""
        c = self.c
        slot = self.slot_ids(x)                            # (B,L,R)
        B, L, R = slot.shape
        live = self.alive[slot]                            # (B,L,R)
        any_live = live.any(-1)                            # (B,L)
        if bool(any_live.any()):       # ★ 回退: 缓存 alive 标志会因直接改 alive 而失效 ⇒ 梯度断
            q = self.ngram_query(x, emb, None, None)       # (B*L,R,d) 全位置
            k = self.key(slot)                             # (B,L,R,d)
            v = self.val(slot)                             # (B,L,R,d)
            logits = -c.beta * (q.view(B, L, R, c.d) - k).pow(2).sum(-1)
            logits = logits.masked_fill(~live, float("-inf"))
            att = torch.nan_to_num(logits.softmax(-1), nan=0.0)
            mask = any_live.unsqueeze(-1).to(att.dtype)    # (B,L,1)
            v_out = v * mask.unsqueeze(-1)                 # 无候选的位置 ⇒ 0
            w_out = att * mask
        else:
            v_out = torch.zeros(B, L, R, c.d, device=x.device,
                                dtype=self.val.weight.dtype)
            w_out = torch.zeros(B, L, R, device=x.device,
                                dtype=self.val.weight.dtype)
        return v_out, w_out, ~any_live

    @torch.no_grad()
    def export_rules(self, topk: int = 20) -> List[Dict]:
        """可解释性: 导出存活槽位的键范数/值范数 (规则向量), 按值范数排序。"""
        idx = self.alive.nonzero().flatten()
        if idx.numel() == 0:
            return []
        vn = self.val.weight[idx].norm(dim=-1)
        order = torch.argsort(vn, descending=True)[:topk]
        return [dict(slot=int(idx[i]), val_norm=float(vn[i]),
                     key_norm=float(self.key.weight[idx[i]].norm())) for i in order]


# --------------------------------------------------------------------- 规则注意力

class RuleAttention(nn.Module):
    """★ v1.2: **SwiGLU 聚合替代 softmax**（桃令）。

    旧版 α=softmax(score) 会物化 (B,H,L,L) 并在其上做 softmax 与 A·V —— 即使内容门关死，
    这两步仍全量跑（实测 100% 关门也只 0.98×）。新版改成：
      ① 规则桶偏置 (B,L,nb) → 选 Kb 个桶 → 每个桶对应**一段相对偏移** → 得到位置**窗口** W=Kb·step
      ② 只在窗口内 gather k,v（O(L·W)，无 (L,L)）
      ③ SwiGLU 门控: gate=SiLU(Wg·k + gc·Wq·q), up=Wu·v, o=Σ_w gate⊙up / √W
    没有 softmax，没有归一化注意力矩阵；"看哪里"由规则决定，"看多少"由 SwiGLU 门决定。
    """

    def __init__(self, c: RCfg):
        super().__init__()
        self.c = c
        hd, dg = c.head_dim, c.gate_rank      # ★ 路线1: 低秩门控
        self.q = nn.Linear(c.d, c.n_heads * hd, bias=False)
        self.k = nn.Linear(c.d, c.n_heads * hd, bias=False)
        self.v = nn.Linear(c.d, c.n_heads * hd, bias=False)
        self.o = nn.Linear(c.n_heads * hd, c.d, bias=False)
        # SwiGLU 聚合 (替代 softmax): 门由 **键** 与 **查询内容** 共同决定
        self.wg = nn.Linear(hd, dg, bias=False)
        self.wq = nn.Linear(hd, dg, bias=False)
        self.wu = nn.Linear(hd, dg, bias=False)
        self.wd = nn.Linear(dg, hd, bias=False)
        # 规则侧
        self.bucket_proj = nn.Linear(c.d, c.n_buckets, bias=True)
        self.gate = nn.Parameter(torch.ones(c.d))
        self.direct = nn.Linear(c.d, c.d, bias=False)        # §4.4 直接路径 (可加)
        self.content_gate = nn.Linear(c.d, 1)                # gc: 查询内容对门的贡献量
        self.pos_abs = nn.Embedding(c.pos_abs_max, dg)       # ★ 显式绝对位置 → 门的查询侧
        # ★ 路线①(b): 位置(可数统计) → **桶偏置**(窗口起点)。门的查询侧(pos_abs)只决定"窗内怎么挑",
        #   到不了"往回看多远"这一环 —— 而 §25 判定缺的正是这一环。零初值 ⇒ 串上后行为逐位不变。
        self.pos_bucket = nn.Embedding(c.pos_abs_max, c.n_buckets)
        nn.init.zeros_(self.pos_bucket.weight)
        nn.init.zeros_(self.content_gate.weight)
        nn.init.constant_(self.content_gate.bias, 4.0)
        self.head_w = nn.Parameter(torch.ones(c.n_cand))
        with torch.no_grad():           # 近端偏置初值 (训练可覆盖)
            b0 = torch.arange(c.n_buckets, dtype=torch.float32)
            self.bucket_proj.bias.copy_(-c.near_first * b0 / max(c.n_buckets - 1, 1))
        self.rule_scale = nn.Parameter(torch.tensor(c.rule_scale_init))
        self.lam = nn.Parameter(torch.tensor(c.lam_init))
        self.k_buckets = c.k_buckets
        # 状态
        self.last_stats: Dict[str, float] = {}
        self.last_agg = None
        self.last_bias = None
        self.last_window = 0
        self.last_startb = None      # 诊断: 动词位置的窗口起点 (规则学出来的那个数)
        self.last_window_off = None  # 诊断: 实际偏移 (夹紧后, B,L,W) —— C22/C28 读它而不是复算公式
        self.last_contrib = None     # 诊断: 各窗口位置的贡献量 (B,L,W)
        self.record_gc = False
        self.last_gc = None
        self.last_gc_t = None
        self.ablate_rule = False
        self.ablate_window = False   # 审计: 只关窗口聚合, 保留 §4.4 direct
        self.ablate_direct = False   # 审计: 只关 §4.4 direct, 保留窗口聚合
        self.record = False
        self.last_mag = {}

    # --- 位置窗口: 选中的桶 → 相对偏移集合 ---
    def _bucket_step(self, L):
        return self.c.bucket_step if self.c.bucket_step > 0 else max(1, -(-L // self.c.n_buckets))

    def _window_index(self, start_bucket, L, pos=None):
        """★ 窗口 = **以偏置最高的桶为起点、长度固定 window_max 的连续偏移区间**。
        (第 1 版是"每选中桶各取冗余 per 个偏移", 结果桶 0-3 只覆盖偏移 0-6, 漏掉距离 7 —— C20 抓出。
         连续区间还有个语义好处: 规则只决定"往回看多远", 窗口内由 SwiGLU 门挑位置。)
        ★ C28 因果不变量: 起点偏移夹到 max(0, 查询位置-W+1)。当可用前缀≥W 时, 整个窗口都在因果边界内;
        前缀短于 W 时, 从偏移 0 开始并只留下已有上下文。旧行为可让窗口整段 j<0 ⇒ vmask 全 0 ⇒
        窗口分子与窗口分支梯度同时为 0 (direct(agg) 是独立旁路)。
        start_bucket: (B,L) 整数; pos: (L,) 或 (1,L)/(B,L) 查询绝对位置 (默认 arange);
        返回 (位置索引 (B,L,W), 有效性 (B,L,W), 实际偏移 (B,L,W))。"""
        c = self.c
        B = start_bucket.shape[0]
        step = self._bucket_step(L)
        W = c.window_max
        off0 = start_bucket * step                       # (B,L) 往回看多远
        if c.clamp_window:
            if pos is None:
                pos = torch.arange(L, device=start_bucket.device).view(1, L)
            pos = pos.to(device=start_bucket.device, dtype=off0.dtype)
            # 窗口偏移为 [off0, off0+W-1]; 使最远项 ≤ query_pos。
            max_start = (pos - W + 1).clamp_min(0)
            off0 = torch.minimum(off0, max_start)
        off = off0.unsqueeze(-1) + _arange(W, start_bucket.device).view(1, 1, W)
        ii = _arange(L, start_bucket.device).view(1, L, 1)
        j = ii - off
        valid = j >= 0
        return j.clamp(0, L - 1).reshape(B, L, W), valid.reshape(B, L, W), off

    def forward(self, h, pos, rule_vec, rule_w, skip, causal=True, has_rule=None):
        c = self.c
        B, L, _ = h.shape
        q = rope(self.q(h).view(B, L, c.n_heads, c.head_dim), pos, c.rope_dim)
        k = rope(self.k(h).view(B, L, c.n_heads, c.head_dim), pos, c.rope_dim)
        v = self.v(h).view(B, L, c.n_heads, c.head_dim)                # (B,L,H,D)

        # ---- 规则路径 → 分桶偏置 (B,L,nb)；全空的位置不参与 ----
        # ★ §51 性能: 原实现是 `int(keep.sum())` + 5 处布尔掩码索引(rule_vec[keep] /
        #   rule_w[keep] / pi[keep] / bias[keep]= / out_direct[keep]=)。CUDA 上布尔掩码
        #   索引要走 nonzero ⇒ 每处一次 device→host 同步, 实测每层 6 次。
        #   改成"全量算 + keep 掩码": 跳过位的 rule_vec/rule_w 本来就都是 0(read 里
        #   置的), 所以乘掩码与"先挑出来再算"逐位等价; bias/out_direct 则用 torch.where
        #   把跳过位钉成 0 —— 与旧版散射后的结果一致。判据 check_fast_path_equiv.py 的
        #   E2(含"漏掉 keep 掩码"的负向对照)。
        keep = ~skip
        # ★ has_rule: "这一批位置里有没有一个带候选的"。**由模型层只判一次**传进来 ——
        #   自己判就是每层一次 int(keep.sum()) 同步。None 表示"调用方没给"(直接单测
        #   RuleAttention 时), 此时退回旧口径自己判, 行为与优化前一致。
        if has_rule is None:
            has_rule = int(keep.sum()) > 0
        if has_rule:
            keepv = keep.unsqueeze(-1)
            hw = self.head_w.view(1, 1, -1, 1)
            g = torch.sigmoid(self.gate * rule_vec)                       # (B,L,R,d)
            agg = (hw * g * rule_vec * rule_w.unsqueeze(-1)
                   * keepv.unsqueeze(-1).to(h.dtype)).sum(2)             # (B,L,d)
            bias_full = self.bucket_proj(agg)                             # 归一化前的**线性**偏置
            if c.use_pos_bucket:
                # ★ 路线①(b): 位置(可数统计) 直接进桶偏置 ⇒ 窗口**起点**能随"数了多少步"移动。
                #   为什么需要这条: §25 判定起点只由查询的**局部 n-gram** 决定, 而"中心词有多远"
                #   (=1+3k) 不在局部 n-gram 里 ⇒ 规则没有信息把窗口放到正确距离 (起点只能塌成常数)。
                pi = pos.clamp(max=c.pos_abs_max - 1).view(1, L).expand(B, L)
                bias_full = bias_full + self.pos_bucket(pi).to(bias_full.dtype)
            zero = torch.zeros((), device=h.device, dtype=h.dtype)
            bias = torch.where(keepv, bias_full.to(h.dtype), zero)
            out_direct = torch.where(keepv, self.direct(agg).to(h.dtype), zero)
            # ★ 契约变更(§51): last_bias 现在是**全量** (B,L,nb), 跳过位显式为 0 ——
            #   正是旧版 `bias[keep] = bias_kept` 散射之后那个 bias。消费方
            #   audit_v11_config 已同步改; last_agg 同理变成全量 (B,L,d)。
            self.last_agg, self.last_bias = agg.detach(), bias.detach()
        else:
            # 全空槽 ⇒ 规则路径整体跳过(§指导书), bucket_proj/direct 一次都不调(C2a 判据)
            bias = torch.zeros(B, L, c.n_buckets, device=h.device, dtype=h.dtype)
            out_direct = torch.zeros(B, L, c.d, device=h.device, dtype=h.dtype)
            self.last_agg, self.last_bias = None, None
        if self.ablate_rule:
            bias = torch.zeros_like(bias)
            out_direct = torch.zeros_like(out_direct)
        if self.ablate_direct:
            # ★ 消融的**细分**: 规则路径有两条腿 —— ① 窗口偏置(决定"看哪里") ② §4.4 直接路径
            #   (h += direct(agg), **不过窗口**)。只关后者才能把"窗口放对了"与"直接路径兜底"分开。
            out_direct = torch.zeros_like(out_direct)

        # 尺度归一化 (审计: 规则分曾被内容分淹没 47 倍), 剖面形状不变
        bias_n = (bias - bias.mean(-1, keepdim=True)) / (bias.std(-1, keepdim=True) + 1e-5)
        bias_n = bias_n * self.rule_scale
        startb = bias_n.argmax(-1)                                            # (B,L) 窗户起点 = 偏置最高的桶
        # ★ 检查 C5 抓出的真问题: 桶偏置若**只**喂 topk().indices, 就是整数索引 ⇒ 不可导
        #   ⇒ rule_scale / bucket_proj 收不到任何梯度, 规则根本没在被学习。
        #   修法: 规则同时给 ① 硬选择(看哪里) ② **可微强度**(选中桶的分值进 SwiGLU 门)。
        # ★ 偏移只算一次: 窗口位置(jidx/jvalid) 与 "每个偏移属于哪个桶"(bidx) 必须**同源**,
        #   否则 C28 的夹紧只作用于位置、而 bsel 仍按未夹紧的偏移取桶 ⇒ 静默不一致。
        self.last_startb = startb.detach() if self.record else None
        jidx, jvalid, off = self._window_index(startb, L, pos)         # (B,L,W)
        W = jidx.shape[-1]
        self.last_window = W
        self.last_window_off = off.detach() if self.record else None    # 诊断: 实际偏移(夹紧后)
        bidx = (off // self._bucket_step(L)).clamp(0, c.n_buckets - 1)
        bsel = bias_n.gather(-1, bidx)          # (B,L,W): gather 要求索引与输入同维数 (可微强度)

        # ---- 窗口 gather (O(L·W), 无 (L,L)) ----
        flat_k = k.reshape(B, L, c.n_heads * c.head_dim)
        bsel_b = torch.arange(B, device=h.device).view(B, 1, 1)   # ★ 必须 (B,1,1): (B,1) 的 dim0 会去对 jidx 的 dim1
        kk = flat_k[bsel_b, jidx].reshape(B, L, W, c.n_heads, c.head_dim)   # (B,L,W,H,D)
        vv = v.reshape(B, L, c.n_heads * c.head_dim)[bsel_b, jidx]             .reshape(B, L, W, c.n_heads, c.head_dim)
        vmask = jvalid.view(B, L, W, 1, 1).to(h.dtype)

        # ---- SwiGLU 聚合 (替代 softmax) ----
        gc = torch.sigmoid(self.content_gate(h))                        # (B,L,1)
        self.last_gc_t = gc.float().mean() if self.record_gc else None
        self.last_gc = float(self.last_gc_t) if self.record_gc else None
        # ★ 门的查询侧加**显式绝对位置** (RoPE 只给相对位置, 解不出 k ⇒ 双峰)
        gq = self.wq(q)
        if c.use_pos_abs:
            gq = gq + self.pos_abs(pos.clamp(max=c.pos_abs_max - 1)).unsqueeze(1)  # (L,1,dg)
        gq = gq.unsqueeze(2) * (gc * self.lam).view(B, L, 1, 1, 1)
        bterm = bsel                                                    # (B,L,W) 规则强度
        gate = F.silu(self.wg(kk) + gq + bterm.view(B, L, W, 1, 1))     # (B,L,W,H,D)
        up = self.wu(vv)                                                # (B,L,W,H,D)
        if c.local_norm:
            # ★ 路线3: 窗口内**逐通道**归一化 —— 把 softmax 的"凸组合/幅度受控"性质拿回来,
            #   代价 O(W) 而非 O(L) (不用整张 (L,L))。
            # ★ 分母必须**只含有效位置** (C29): 旧写法 gate.abs().sum(2) 把越界位置也计入分母
            #   ⇒ 输出随"窗口里有几个越界位置"缩放, 且越界位置(分子贡献恒 0)仍经分母产生梯度。
            gsum = (gate.abs() * vmask).sum(2)                             # (B,L,H,Dg)
            o = (gate * up * vmask).sum(2) / (gsum + 1e-5)
        else:
            o = (gate * up * vmask).sum(2) / math.sqrt(max(W, 1))       # (B,L,H,D) 窗口内求和
        if self.record:
            # 诊断读数: 每个**窗口位置**对输出的贡献 (对 (头,通道) 求和) —— 可失败(会随输入变), 见 C23
            self.last_contrib = (gate * up * vmask).sum(dim=(3, 4)).detach()
        o = self.wd(o).reshape(B, L, c.n_heads * c.head_dim)
        if self.ablate_window or self.ablate_rule:
            o = torch.zeros_like(o)
        if self.record:
            self.last_mag = dict(rule_absmean=float(bias_n.abs().mean()),
                                 window=W, gate_mean=float(gate.mean()),
                                 rule_scale=float(self.rule_scale), lam=float(self.lam),
                                 head_w_mean=float(self.head_w.mean()))
        o = o * keep.unsqueeze(-1).to(o.dtype)      # 无候选位置 → 0 (残差捷径 §4.5)
        # ★ §51: rule_skip_ratio 改成**惰性张量**, 前向里不再 int() ⇒ 每层少一次同步。
        #   物化推迟到 stats() 被调用时(调用点是日志/审计, 不是每步)。
        self.last_stats = dict(total=B * L,
                               rule_skip_ratio_t=1.0 - skip.to(torch.float32).mean())
        return self.o(o) + out_direct


class RuleBlock(nn.Module):
    """残差 + 混合注意力 (+ 可选 FFN)。无触发时规则路径为零 ⇒ h ≈ h (§4.5)。"""

    def __init__(self, c: RCfg):
        super().__init__()
        self.c = c
        self.norm1 = nn.LayerNorm(c.d)
        self.attn = RuleAttention(c)
        self.norm2 = nn.LayerNorm(c.d)
        self.ffn = nn.Sequential(nn.Linear(c.d, c.ffn_mult * c.d), nn.GELU(),
                                 nn.Linear(c.ffn_mult * c.d, c.d))
        self.use_ffn = True

    def forward(self, h, pos, rule_vec, rule_w, skip, has_rule=None):
        h = h + self.attn(self.norm1(h), pos, rule_vec, rule_w, skip, has_rule=has_rule)
        if self.use_ffn:
            h = h + self.ffn(self.norm2(h))
        return h


class EngramRuleLM(nn.Module):
    def __init__(self, c: RCfg):
        super().__init__()
        self.c = c
        # ★★ §61: `use_qpool` 必须在**任何 nn.Module/Parameter 构造之前**设好。
        #   池化参数放在 emb **之前**建, 但 `nn.Linear` 的构造**本身会消耗 RNG**
        #   (kaiming_uniform 初始化), 这会让后面所有模块的初值与旧版不同 ——
        #   所以建完立刻把 seed 恢复到调用方设的值。
        #   ★调用方契约: **必须先 `torch.manual_seed(seed)` 再构造本模型**,
        #     否则无 seed 可恢复(旧版从不依赖这一点, 现在也不该依赖)。
        #   ⇒ 这两条合起来保证 `use_qpool=False` 与 `True+q_pool=None`
        #     的输出与旧版**逐位相同**(判据 check_qpool.py 的 Q2)。
        _saved_seed = torch.initial_seed()
        self.use_qpool = bool(c.use_qpool)
        if self.use_qpool:
            self.qpool_alpha = nn.Parameter(torch.tensor(float(c.qpool_alpha_init)))
            self.qpool_proj = nn.Linear(c.d, c.d, bias=False)
            nn.init.eye_(self.qpool_proj.weight)      # 初值为恒等 ⇒ 初始等价于"原样加"
            torch.manual_seed(_saved_seed)            # ★把 RNG 还回去(见上)
        self.q_pool = None
        self._q_pool_arg = None          # forward(q_pool=...) 的当次覆盖, 不落盘
        self.emb = nn.Embedding(c.vocab, c.d)
        self.engram = EngramMemory(c)
        self.blocks = nn.ModuleList([RuleBlock(c) for _ in range(c.n_layers)])
        self.norm = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        # ★不变量: 开/关池化只差**这 2 个**参数, 其余逐位相同(判据 Q2)。
        #   这条断言让"改通路不改初值"在每次构造时自动被检查, 而不只靠
        #   check_qpool.py 单独跑的那一遍。
        # ★个数不写死: 出货配置是 60, 但单测用的小配置是 33 —— 写死会让
        #   check_dialogue_small.py 那种小模型测试直接断言失败(踩过一次)。
        #   改成"开与关必须**只差 2**", 这才是真正想守的不变量。
        _names = {n for n, _ in self.named_parameters()}
        _qpool_names = {"qpool_alpha", "qpool_proj.weight"}
        if self.use_qpool:
            assert _qpool_names <= _names, "池化参数缺失: %s" % sorted(_qpool_names - _names)
        else:
            assert not (_qpool_names & _names), \
                "use_qpool=False 却建了池化参数: %s" % sorted(_qpool_names & _names)
        _extra = len(_names & _qpool_names)
        assert _extra == (2 if self.use_qpool else 0), \
            "池化开关与参数个数不自洽: use_qpool=%s 但多了 %d 个池化参数" \
            % (self.use_qpool, _extra)

    def set_question_pool(self, q_pool):
        """q_pool: (B,d) 或 None。None = 关闭本通路(输出与旧版逐位相同)。

        也可以每次调用 `forward(x, q_pool=...)` 现给 —— 训练时每批的问题不同,
        逐批传入比"设一次"更自然。显式传入的**优先于**这里设的。
        """
        self.q_pool = q_pool

    def _apply_qpool(self, h):
        """把问题摘要加到每个位置。成本 O(B·L·d), 一次广播加法。

        ★α 初始很小(0.1)且可学习 ⇒ 训练自己决定用多少;
        ★投影初值为单位阵 ⇒ 初始时通路只是"把问题均值加进去", 不引入随机扰动。
        """
        q = self._q_pool_arg if self._q_pool_arg is not None else self.q_pool
        if q is None:
            return h
        # ★状态污染守卫: 自动切段只在"本批真有 [USER] 且 cfg 给了 id"时产出 q_pool。
        #   若上一批算过、这一批没有, `_q_pool_arg` 会**残留**上一批的值
        #   (踩过: 评测时混批, 无问题的行被喂了别人的问题摘要)。
        #   做法: 记住"这个 q_pool 是为哪个 batch 形状/哪一行算的", 形状不符就作废。
        if q.shape[0] != h.shape[0]:
            raise RuntimeError(
                "q_pool 的 batch(%d) 与本批 h(%d) 不符 —— 疑似上一批残留。"
                "请改用 forward(x, q_pool=...) 逐批传入。"
                % (q.shape[0], h.shape[0]))
        if q.shape[-1] != h.shape[-1]:
            raise RuntimeError(
                "q_pool 形状 %s 与 h 的 (B=%d,d=%d) 不匹配"
                % (tuple(q.shape), h.shape[0], h.shape[-1]))
        return h + self.qpool_alpha * self.qpool_proj(q).unsqueeze(1)

    def forward(self, x, q_pool=None):
        """q_pool 传 None 时, 若 cfg 给了 [USER]/[ASSISTANT] 的 id 就**自己切**问题段。

        ★为什么要自己切: 训练时每行的问题不同, 调用方逐个算 (B,d) 既慢又要重复写
          一遍切段逻辑。放在模型里做一次 gather, 实测 0.1 ms vs 逐行 6.8 ms。
        ★向后兼容: 显式传的 q_pool **优先**; cfg 没给 id 时退化为"必须显式传",
          与 §61 最初版行为一致。
        ★★`_q_pool_arg` 是**当次**覆盖, 每次 forward 开头先清成 None ——
          否则上一批算出的 q_pool 会残留到下一批(形状不同的批次会抛, 同形状则静默串味)。
        """
        c = self.c
        B, L = x.shape
        self._q_pool_arg = q_pool          # ★每次 forward 先重置(见上)
        if q_pool is None and self.use_qpool and c.qpool_user_id >= 0:
            u, a = c.qpool_user_id, c.qpool_assist_id
            is_u = (x == u)
            is_a = (x == a)
            # 每个 batch 取**最后一次**出现的 [USER] 与**第一次**出现在它之后的 [ASSISTANT]
            if bool(is_u.any()) and bool(is_a.any()):
                pos = torch.arange(L, device=x.device).view(1, L).expand(B, L)
                u_pos = torch.where(is_u, pos, torch.full_like(pos, -1)).max(dim=1).values
                a_pos = torch.where(is_a, pos, torch.full_like(pos, L + 1)).min(dim=1).values
                in_q = ((pos > u_pos.view(-1, 1)) & (pos < a_pos.view(-1, 1))
                        & (u_pos.view(-1, 1) >= 0))
                cnt = in_q.sum(1).clamp_min(1)
                emb = self.emb(x)                                   # (B,L,d)
                q_pool = (emb * in_q.unsqueeze(-1).to(emb.dtype)).sum(1) / cnt.view(-1, 1)
                # 没有问题 token 的行(如纯历史)⇒ 池化退化为 0, 等价于"没这通路"
                q_pool = q_pool * (cnt > 1).view(-1, 1).to(q_pool.dtype)
        self._q_pool_arg = q_pool          # 供 _apply_qpool 取用(不落盘)
        pos = torch.arange(L, device=x.device)
        h = self.emb(x)
        rule_vec, rule_w, skip = self.engram.read(x, self.emb)
        # ★ §51: "这一批位置有没有带候选的" **只判一次**(一次同步), 传给所有层 ——
        #   否则每层一次 int(keep.sum())。全空 ⇒ 规则路径整体跳过, 与优化前同。
        has_rule = bool((~skip).any())
        for blk in self.blocks:
            h = blk(h, pos, rule_vec, rule_w, skip, has_rule=has_rule)
            if self.use_qpool:
                h = self._apply_qpool(h)     # 每层之后都注入(与残差同源)
        return self.head(self.norm(h))

    def content_usage(self, grad=False):
        """平均内容门值。grad=True 返回**可微张量** (惩罚项必须用这个版本);
        grad=False 返回 float 仅供报数。"""
        gs = [b.attn.last_gc_t if grad else b.attn.last_gc for b in self.blocks]
        gs = [g for g in gs if g is not None]
        if not gs:
            return torch.tensor(1.0, device=next(self.parameters()).device) if grad else 1.0
        return torch.stack(gs).mean() if grad else float(sum(gs) / len(gs))

    def set_sparse_content(self, on, tau=0.05):
        for b in self.blocks:
            b.attn.sparse_content = on
            b.attn.gate_tau = tau

    # ---------- 统计 / 可解释 ----------
    def stats(self) -> Dict[str, float]:
        """★ §51: rule_skip_ratio 在前向里是**惰性张量**(避免每层一次 D2H 同步),
        在这里才 float() 物化。float(张量) 本身是一次同步 —— 所以只许在日志/审计
        这类低频调用点用, 别挪回前向。"""
        s = self.blocks[0].attn.last_stats if self.blocks else {}
        ratio = s.get("rule_skip_ratio", None)
        if ratio is None:
            lazy = s.get("rule_skip_ratio_t", None)
            ratio = float(lazy) if lazy is not None else float("nan")
        return dict(rule_skip_ratio=ratio,
                    alive_slots=int(self.engram.alive.sum()),
                    n_slots=self.c.n_slots)

    def export_rules(self, topk: int = 20):
        return self.engram.export_rules(topk)

    def flops_per_token(self, L: int) -> Dict[str, float]:
        """稠密 vs 实际 (稀疏触发后) 的每 token FLOPs 估算 —— 用于"加速比"。"""
        d, H = self.c.d, self.c.n_heads
        dense_attn = 2 * L * d * 2                     # qk + Av, 每头合计按 d 计
        content = 2 * L * d * 2
        rule = 2 * self.c.n_cand * d                   # 每位置 R 次 d 维点积 (不含 L 项!)
        return dict(dense_attn_per_tok=dense_attn, content_per_tok=content,
                    rule_per_tok=rule,
                    speedup_upper_bound=dense_attn / max(rule, 1e-9))
