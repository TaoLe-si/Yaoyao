"""只读审计: 核实 DeepSeek 对 v11 架构的三条断言。

被检验的断言(逐条给出可证伪预测):
  A1 "当前模型参数是 4.8M"                -> 预测: 与 checkpoint 实测不符。
  A2 "argmax 硬选择 => 规则层收不到梯度"   -> 预测: bucket_proj / pos_bucket 梯度恒为 0。
  A3 "架构缺少数前缀的通路"                -> 预测: use_pos_bucket 关闭 / pos_bucket 未接线。

负向对照(判据可失败性): 同一指标必须在**换输入后改变**。
  - 若两次不同输入得到逐位相同的梯度范数, 该量是常量, 按 AGENTS.md 规则 7 不算证据。
  - 若某张量梯度为 0, 再用"人为把该通路置为非零"证明该指标确实会动。

只读: 不写任何 checkpoint / 报告。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")


def grad_norm(p):
    return float(p.grad.abs().sum()) if p.grad is not None else float("nan")


def build_engram_from_examples(model, x, lengths):
    """与 dialogue_small.py:345 同逻辑的内联副本 (避免为只读审计引入 tokenizers/requests)。"""
    counts = torch.zeros(model.c.n_slots, dtype=torch.long, device=model.engram.alive.device)
    for i, length in enumerate(lengths.tolist()):
        if length < 2:
            continue
        ids = model.engram.slot_ids(x[i:i + 1, :length].to(model.engram.alive.device)).reshape(-1)
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.long))
    model.engram.alive.copy_(counts >= model.c.min_count)


def run(seed, pos_bucket_random=False):
    """构造 v11 配置 + 随机权重, 跑一次前向/反向, 返回各参数梯度范数与选桶分布。

    pos_bucket_random=True 是 **负向对照**: v11 出厂 pos_bucket 是零初始化
    (model.py:290), 此时 bsel 在窗口内恒为常数 —— 按 AGENTS.md 规则 7, 常量不算证据。
    只有把它随机化后 bsel_spread 变成非零, 才能说"零初始化下它确实不会变"。
    """
    from model import RCfg, EngramRuleLM

    torch.manual_seed(seed)
    cfg = RCfg(vocab=1024, d=64, n_layers=2, n_heads=4, ffn_mult=2, rope_dim=8,
               ngram_max=2, n_hash=2, n_slots=1 << 12, min_count=1, n_buckets=16,
               gate_rank=16, pos_abs_max=512, use_pos_bucket=True, window_max=16,
               bucket_step=32)
    model = EngramRuleLM(cfg)
    if pos_bucket_random:
        with torch.no_grad():
            for blk in model.blocks:
                blk.attn.pos_bucket.weight.normal_(0, 1.0)
    n_param = sum(p.numel() for p in model.parameters())

    g = torch.Generator().manual_seed(seed)
    B, L = 4, 512
    x = torch.randint(0, 1024, (B, L), generator=g)
    lengths = torch.full((B,), L, dtype=torch.long)
    build_engram_from_examples(model, x, lengths)
    attn = model.blocks[0].attn
    attn.record = True                      # 否则 last_startb 为 None (诊断读数被自己关掉)

    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = out.float().pow(2).mean()
    loss.backward()

    attn = model.blocks[0].attn
    _, _, skip = model.engram.read(x, model.emb)
    bfrac, bmax = _bsel_spread(attn, x, skip)
    return dict(
        n_param=n_param,
        loss=float(loss.detach()),
        bucket_proj=grad_norm(attn.bucket_proj.weight),
        pos_bucket=grad_norm(attn.pos_bucket.weight),
        rule_scale=grad_norm(attn.rule_scale),
        direct=grad_norm(attn.direct.weight),
        startb_mean=float(attn.last_startb.float().mean()) if attn.last_startb is not None else float("nan"),
        # 结构性读数: 窗口内 bsel (=bias_n.gather(off//step)) 是否随窗口内位置变化。
        # v11 的 bucket_step=32, window_max=16 => 窗口横跨 16 个偏移, 落在 1~2 个桶块内。
        bsel_spread_frac=bfrac, bsel_spread_max=bmax,
    )


def _bsel_spread(attn, x, skip):
    """返回 (窗口内 bsel 有空间差异的位置占比, 最大 spread)。
    占比>0 => 规则强度在窗口内逐位置区分; 占比=0 => 窗口内是**空间均匀标量**。"""
    if attn.last_bias is None:
        return float("nan"), float("nan")
    L = x.shape[1]
    with torch.no_grad():
        # ★ §51 契约变更: last_bias 现在**就是**全量 (B,L,nb), 跳过位显式为 0
        #   (旧版只含 keep 位, 需要这里手工散射回去)。
        bias = attn.last_bias.detach().to(torch.float32)
        bias_n = (bias - bias.mean(-1, keepdim=True)) / (bias.std(-1, keepdim=True) + 1e-5)
        bias_n = bias_n * float(attn.rule_scale)
        startb = bias_n.argmax(-1)
        jidx, jvalid, off = attn._window_index(startb, L, None)
        step = attn._bucket_step(L)
        bidx = (off // step).clamp(0, attn.c.n_buckets - 1)
        bsel = bias_n.gather(-1, bidx)                      # (B,L,W)
        spread = (bsel.max(-1).values - bsel.min(-1).values).abs()
        return float((spread > 1e-9).float().mean()), float(spread.max())


def main():
    print("checkpoint:", os.path.basename(CKPT), "exists=", os.path.exists(CKPT))
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg")
        print("\n[A1] checkpoint 内保存的 cfg:")
        for k in ("vocab", "d", "n_layers", "n_heads", "rope_dim", "ffn_mult",
                  "n_buckets", "window_max", "bucket_step", "gate_rank",
                  "use_pos_bucket", "use_pos_abs", "pos_abs_max", "local_norm",
                  "clamp_window", "near_first"):
            print("   %-14s = %s" % (k, cfg.get(k, "<缺>") if isinstance(cfg, dict) else "<缺>"))
        sd = ck.get("state_dict")
        print("   state_dict 张量数 =", len(sd))
        print("   state_dict 参数总数 =", sum(v.numel() for v in sd.values()))

    print("\n[A2/A3] 梯度可失败性检查 (CPU, 两次不同输入):")
    a = run(1)
    b = run(2)
    print("   参数总数 seed1 =", a["n_param"], " seed2 =", b["n_param"])
    for k in ("loss", "bucket_proj", "pos_bucket", "rule_scale", "direct", "startb_mean",
              "bsel_spread_frac", "bsel_spread_max"):
        va, vb = a[k], b[k]
        same = (va == vb)
        print("   %-12s seed1=%-14.6g seed2=%-14.6g  %s"
              % (k, va, vb, "常量(不可失败!)" if same else "随输入改变"))

    print("\n   判决: bucket_proj 梯度 %s ; pos_bucket 梯度 %s"
          % ("非零" if a["bucket_proj"] > 0 else "为零",
             "非零" if a["pos_bucket"] > 0 else "为零"))

    print("\n[负向对照] 把 pos_bucket 随机化(出厂为零初始化)后, bsel 是否会在窗口内变化:")
    c1 = run(1, pos_bucket_random=True)
    print("   startb_mean    = %.4f" % c1["startb_mean"])
    print("   bsel_spread_frac = %.4f" % c1["bsel_spread_frac"])
    print("   bsel_spread_max  = %.6g" % c1["bsel_spread_max"])
    if c1["bsel_spread_frac"] > 0:
        print("   => 判据可失败: 零初始化下恒为 0 是**配置**的结果, 不是指标写死。")
    else:
        print("   => ★判据不可失败, bsel_spread 这条证据作废 (先读检查本身)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())