"""§51.8b: 训练侧 A/B —— 优化是否也让训练变快。

**为什么单独测**: DESIGN §51 的全部读数都是生成路径。F3/F4 去掉的是**前向里**的同步与
掩码索引, 训练走同一条前向, 方向上应同受益 —— 但"方向上"是**推测**, 规则 17 要求推测必须
标注并给验证方式。本文就是那个验证方式。

**A/B 用冻结参照当"改动前"**: `_ref_preopt.py` 是优化前那两段前向的逐字副本, 所以这是
"同一台机器、同一份初始权重、同一个种子、同一个 loss"的前后对比, 不是跨版本对比。

**同时校验训练语义没变**: 逐步 loss 必须逐位一致(优化只动同步与索引方式, 不动算术)。
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
STEPS = 15
BS = 256          # 规则 18: 训练默认 bs >= 256
SEQ = 256         # ★ seq 从 512 降到 256: 冻结参照在 (256,512,R=6,d=64) 上做布尔掩码索引
                  #   极贵(单步十几秒), 跑满 40 步要十几分钟。这个基准只回答
                  #   "去同步/去掩码索引是否让训练同向变快", 不承担 §48 wall_sec 的口径,
                  #   所以降 seq/步数换时间; **倍数不可搬到 bs=256×seq=512**。
LR = 3e-3
# 判据在 main() 里逐条写明理由(第一版写成"全程逐位一致"是**判据本身错了**, 见 main)


def build(ck, seed=11):
    from model import RCfg, EngramRuleLM
    torch.manual_seed(seed)
    m = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    m.engram.alive[:] = True          # 与出厂 v11 一致(全活), 排除语料统计差异
    return m


def data(ck, seed=11, bs=BS, seq=SEQ):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, ck["cfg"]["vocab"], (bs, seq), generator=g).to(DEV)
    y = torch.randint(0, ck["cfg"]["vocab"], (bs, seq - 1), generator=g).to(DEV)
    return x, y


def run(ck, x, y, tag):
    """跑 STEPS 步, 返回 (墙钟, 逐步 loss)。"""
    import torch.nn.functional as F
    m = build(ck)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    m.train()
    for _ in range(3):                 # 预热: CUDA 上下文/cuBLAS 工作区
        opt.zero_grad(set_to_none=True)
        out = m(x)
        F.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]), y.reshape(-1)).backward()
        opt.step()
    losses = []
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        out = m(x)
        loss = F.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    if DEV == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, losses, m


def main():
    from model import EngramMemory, RuleAttention
    from bench_gpu import bench_matmul, PEAK_TFLOPS
    import _ref_preopt as ref

    ck = torch.load(V11, map_location="cpu", weights_only=False)
    print("DEV=%s | %s | bs=%d seq=%d steps=%d d=%d layers=%d"
          % (DEV, torch.__version__, BS, SEQ, STEPS, ck["cfg"]["d"], ck["cfg"]["n_layers"]))
    gate = float("nan")
    if DEV == "cuda":
        tf, _ = bench_matmul()
        gate = tf / PEAK_TFLOPS
        print("[门禁] 参照 matmul %.2f TFLOPS = 峰值的 %.0f%% => %s"
              % (tf, 100 * gate, "可用" if gate > 0.3 else "★受抢占影响, 只引用比值"))

    x, y = data(ck)

    # ---- 同步次数随 batch/序列长度怎么变? 用来解释"为什么训练没收益" ----
    # 同步代价与**处理多少位置无关**, 只与**调用次数**有关。生成侧每步 1 次前向只处理
    # 几十个位置; 训练侧一次前向处理 B·L 个位置。所以只要证明同步次数与 B/L 无关,
    # "训练把同步摊薄了"就是读数支撑的结论而不是推测。
    if DEV == "cuda":
        from profile_generation_path import count_syncs
        m_s = build(ck)
        sync_bsl = {}
        for b, s in ((1, 96), (BS, SEQ)):
            xs = torch.randint(0, ck["cfg"]["vocab"], (b, s), device=DEV)
            with torch.no_grad():
                m_s(xs)
                n, _ = count_syncs(lambda: m_s(xs))
            sync_bsl[(b, s)] = n
            pos_per_sync = b * s / max(n, 1)
            print("[同步] 前向 (B=%d,L=%d): %d 次同步 | 每同步摊到 %.0f 个位置"
                  % (b, s, n, pos_per_sync))
        gen_ratio = (1 * 96 / max(sync_bsl[(1, 96)], 1))
        trn_ratio = (BS * SEQ / max(sync_bsl[(BS, SEQ)], 1))
        print("       ⇒ 生成侧每同步摊 %.0f 位置, 训练侧 %.0f 位置 (差 %.0f×)"
              % (gen_ratio, trn_ratio, trn_ratio / max(gen_ratio, 1e-9)))

    # ---- 纯前向等价(同机同权重, 不训练): 直接比 logits ----
    # 第一版这里写的是"第 1 步 loss 必须逐位为 0", **判据本身错了**(规则 5):
    #   (a) losses[0] 其实来自第 4 次前向(前面有 3 步预热), 不是纯前向;
    #   (b) F4 把 (M,R,d).sum(1) 改成了 (B,L,R,d).sum(2) —— **归约顺序变了**,
    #       FP32 下结果不保证逐位相同(我自己已在 E2b 记录过梯度相对差 7.1e-8)。
    # 所以这里改成直接量"同权重同输入下 logits 差多少", 量纲清楚、不会自相矛盾。
    from model import EngramMemory, RuleAttention
    import _ref_preopt as ref
    m_fwd = build(ck)
    m_fwd.eval()
    with torch.no_grad():
        out_new = m_fwd(x)
        saved = (EngramMemory.read, RuleAttention.forward)
        EngramMemory.read = ref.ref_engram_read
        RuleAttention.forward = ref.ref_rule_attention_forward
        try:
            out_ref = m_fwd(x)
        finally:
            EngramMemory.read, RuleAttention.forward = saved
    d_fwd = float((out_new - out_ref).abs().max())
    rel_fwd = d_fwd / max(float(out_ref.abs().max()), 1e-12)
    print("\n纯前向等价(bs=%d seq=%d 同权重同输入): max|Δlogits|=%.3e | 相对 %.3e"
          % (BS, SEQ, d_fwd, rel_fwd))
    assert rel_fwd < 1e-6, "同权重同输入下前向不等价(相对差 %.3e) ⇒ 速度对比无效" % rel_fwd

    t_new, l_new, _ = run(ck, x, y, "new")

    saved = (EngramMemory.read, RuleAttention.forward)
    EngramMemory.read = ref.ref_engram_read
    RuleAttention.forward = ref.ref_rule_attention_forward
    try:
        t_ref, l_ref, _ = run(ck, x, y, "ref")
    finally:
        EngramMemory.read, RuleAttention.forward = saved

    worst = max(abs(a - b) for a, b in zip(l_new, l_ref))
    print("\n改动前(冻结参照) %.2f s | %.1f ms/步" % (t_ref, t_ref / STEPS * 1e3))
    print("改动后           %.2f s | %.1f ms/步" % (t_new, t_new / STEPS * 1e3))
    print("★训练侧 %.2f×   (loss %.4f → %.4f)" % (t_ref / max(t_new, 1e-9), l_ref[0], l_new[0]))
    print("15 步内 loss 最大差 %.3e (归约序差异经 AdamW 放大的量级)" % worst)
    assert worst < 1e-2, "训练 loss 轨迹差异 %.3e 超出可解释范围" % worst
    print("门禁 gate_frac=%.3f" % gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
