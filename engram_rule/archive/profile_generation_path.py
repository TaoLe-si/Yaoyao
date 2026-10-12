"""§51 生成路径 profile: 只测不改, 定位 "慢得要死" 的开销分布。

**纪律**: AGENTS.md 规则 20 —— 性能结论必须有 profile 读数(CPU ms vs CUDA ms), 不许猜。
本文所有测量都走**真实代码路径**(包装 model.forward / _select_next_id), 不重写一份循环,
以免测的不是线上那条路。

────────────────────────────────────────────────────────────────────────
预登记的可证伪预测 (DESIGN §51)。每条都给"反例下会变"的证据, 不给就当没测。

P1  host 侧 Python 逐 token 循环主导。
    机制: _select_next_id 里 `for token_id in range(scores.numel())` 对 vocab 个位置
    逐个做张量索引赋值 ⇒ 每步 vocab 次 CUDA kernel launch。
    **否证**: select 的 host 时间 < 前向 host 时间的 20% ⇒ P1 被否。
    负向对照: 同一计时器包一个纯 host 的死循环, 读数必须随循环次数变化(否则计时器坏了)。

P2  self.head 对**全部** L 个位置算 logits 是前向 FLOP 的大头。
    机制: head 是 2·B·L·d·V, 而 attn+ffn+engram 合计 ≈ 2·B·L·d·(4d+2·ffn_mult·d),
          V(=vocab) 通常比 (4+2·ffn_mult)·... 大一个量级。
    **否证**: 实测 head 占前向 CUDA 时间 < 40% ⇒ P2 被否。

P3  前向是 launch-bound 而非 FLOP-bound。
    机制: d=64/2层 的 matmul 极小, 单核发不出算力。
    **否证**: t(B=32)/t(B=1) > 8(接近线性加速) ⇒ **P3 被否**(说明本来就吃满算力, 批量化没用)。

P4  前向与选择里存在 device→host 同步, 每步把流水线排空。
    机制: RuleAttention 里 `int(keep.sum())`(每层一次)、EngramMemory.read 里
          `bool(flat.any())`、选择里 `bool(isfinite().any())` 与 `int(argmax())`。
    **否证**: 单次前向的同步计数为 0 ⇒ P4 被否。
    负向对照: 纯异步区段必须报 0; 含 float(t.sum()) 的区段必须报 ≥1 —— 否则计数器不可信。

P5  KV cache 可做到**逐位等价** ⇒ 值得做。
    机制: 窗口偏移由 `start_bucket · _bucket_step(L)` 决定; 若 bucket_step 是**常量**
          (v11 由 dialogue_small.py:1116 显式设为 ceil(max_len/16) > 0), 偏移与 L 无关;
          又 j = pos - off ≤ pos, 即位置 i 的输出只依赖 x[0..i] ⇒ 前缀不变性成立。
    **否证**: (a) 前缀不变性检查不通过; 或 (b) bucket_step == 0(legacy 动态步长)。
    **负向对照**: 把 bucket_step 改成 0 的同一份权重, 同一检查**必须失败** —— 否则检查无效。

P6  ~~生成路径的 GPU 忙时占比高~~ **本条已作废(§51 修订1)**。
    原写法用 CUDA event 的 span 当"GPU 忙时", 这是**无效判据**: span 覆盖的是"流打开到关闭"
    的整段墙钟, host 侧排队等待期间 GPU 照样计入 ⇒ 一个"完全空转"的实现也能测出 100%。
    正确测法见 §F: 用 profiler 的 self CUDA time 求 kernel 真实忙时, 再与 wall 比。
"""
import os
import sys
import time
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_ROWS = 8            # profile 用行数: 够算均值, 又不至于跑几分钟
MAX_NEW = 90          # 与 probe_rerank_mbr.py 同口径
PREFIX_TOL = 1e-4     # 前缀不变性容差(见 §51 正文: FP32 不同 L 的归约序不同, 1 ulp 级差异会放大)


# ----------------------------------------------------------------- 计时器自检

def timeit_selfcheck():
    """负向对照: 计时器必须能分辨不同的 host 耗时量级, 否则后面所有读数都不可信。"""
    out = {}
    for n in (1_000, 8_000):
        t0 = time.perf_counter()
        acc = 0
        for i in range(n):          # 纯 host 死循环, 不碰 GPU
            acc += i % 7
        dt = time.perf_counter() - t0
        out[n] = dt * 1e3
        assert acc >= 0
    ratio = out[8_000] / max(out[1_000], 1e-9)
    ok = ratio > 5.0
    print("[自检] host 计时器 1k=%.3fms 8k=%.3fms 比值=%.1f => %s"
          % (out[1_000], out[8_000], ratio, "PASS" if ok else "FAIL(计时器分辨不出量级)"))
    assert ok, "host 计时器自检失败: 读数不可信"
    return ratio


def count_syncs(fn):
    """数 device→host 同步次数。**必须能失败**: 纯异步区段报 0, 含 .item() 的报 ≥1。"""
    if DEV != "cuda":
        return -1, ["无 GPU, 跳过"]
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    msgs = [str(w.message) for w in caught if "synchron" in str(w.message).lower()]

    # 负向对照 A: 纯异步(只排 kernel, 不读回)必须 0
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as c0:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            a = torch.randn(8, 8, device=DEV)
            (a @ a).sum()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    n_async = len([w for w in c0 if "synchron" in str(w.message).lower()])

    # 负向对照 B: 读回标量必须 ≥1
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as c1:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            float(a.sum())
        finally:
            torch.cuda.set_sync_debug_mode("default")
    n_item = len([w for w in c1 if "synchron" in str(w.message).lower()])
    assert n_async == 0, "同步计数器把纯异步区段也报成 %d 次 => 计数器不可信" % n_async
    assert n_item >= 1, "含 float(t.sum()) 的区段没报出同步 => 计数器不可信"
    return len(msgs), ["async对照=%d(期望0)" % n_async, "item对照=%d(期望>=1)" % n_item]


# ----------------------------------------------------------------- 测量

def profile_generation(model, tok, rows):
    """包住**真实**的 model.forward 与 _select_next_id, 拆出 host 侧耗时; 用 CUDA event
    量整段 GPU span。只读, 不改任何返回值。"""
    import dialogue_small as ds

    stat = {"fwd_host": 0.0, "sel_host": 0.0, "steps": 0}
    orig_fwd = model.forward
    orig_sel = ds._select_next_id

    def timed_fwd(*a, **kw):
        t0 = time.perf_counter()
        r = orig_fwd(*a, **kw)
        stat["fwd_host"] += time.perf_counter() - t0
        stat["steps"] += 1
        return r

    def timed_sel(*a, **kw):
        t0 = time.perf_counter()
        r = orig_sel(*a, **kw)
        stat["sel_host"] += time.perf_counter() - t0
        return r

    model.forward = timed_fwd
    ds._select_next_id = timed_sel
    try:
        if DEV == "cuda":
            torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True) if DEV == "cuda" else None
        ev1 = torch.cuda.Event(enable_timing=True) if DEV == "cuda" else None
        t0 = time.perf_counter()
        if ev0:
            ev0.record()
        outs = [ds.greedy_generate(model, tok, r, max_new_tokens=MAX_NEW, device=DEV)
                for r in rows]
        if ev1:
            ev1.record()
        wall = time.perf_counter() - t0
        if DEV == "cuda":
            torch.cuda.synchronize()
            gpu_ms = ev0.elapsed_time(ev1)
        else:
            gpu_ms = float("nan")
    finally:
        model.forward = orig_fwd
        ds._select_next_id = orig_sel

    other = wall - stat["fwd_host"] - stat["sel_host"]
    return dict(wall_s=wall, gpu_span_s=gpu_ms * 1e-3,
                fwd_host_s=stat["fwd_host"], sel_host_s=stat["sel_host"], other_s=other,
                steps=stat["steps"], n_rows=len(rows),
                gen_tokens=[o["generated_tokens"] for o in outs],
                outputs=[o["generated"] for o in outs]), outs


def batch_scaling(model, L, batch_list):
    """前向随 B 的时间曲线。用来判 launch-bound(P3)。"""
    res = {}
    with torch.no_grad():
        for B in batch_list:
            x = torch.randint(0, model.c.vocab, (B, L), device=DEV)
            for _ in range(3):
                model(x)
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 10
            for _ in range(reps):
                model(x)
            if DEV == "cuda":
                torch.cuda.synchronize()
            res[B] = (time.perf_counter() - t0) / reps * 1e3
    return res


def head_share(model, L):
    """head(self.norm(h)) 的时间占比: 把 head 换成 Identity 再测一遍(只测, 立刻还原)。"""
    with torch.no_grad():
        x = torch.randint(0, model.c.vocab, (1, L), device=DEV)
        for _ in range(3):
            model(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
        reps = 10
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
        full = (time.perf_counter() - t0) / reps * 1e3

        orig = model.head
        model.head = torch.nn.Identity()
        try:
            for _ in range(3):
                model(x)
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
            if DEV == "cuda":
                torch.cuda.synchronize()
            nohead = (time.perf_counter() - t0) / reps * 1e3
        finally:
            model.head = orig
    d, v = model.c.d, model.c.vocab
    ff = 2 * 2 * L * d * (4 * d + 2 * model.c.ffn_mult * d)
    ana = 2 * L * d * v / max(ff, 1)
    return dict(full_ms=full, nohead_ms=nohead, measured_share=1.0 - nohead / max(full, 1e-9),
                analytic_head_over_rest=ana)


@torch.no_grad()
def prefix_invariance(cfg_dict, state, L_full=96, L_cut=60, device=DEV):
    """P5 的可失败检查: bucket_step>0 时, 位置 i 的隐状态只能依赖 x[0..i]。

    **负向对照必须真的会失败**, 否则这个检查不是判据(规则 7)。难点: 训练好的权重里
    bucket_proj.bias 带 near_first 偏置 ⇒ startb 很可能恒为 0 ⇒ 窗口恒从 offset 0 起
    ⇒ 即使 bucket_step 是动态的也测不出差异(检查"永远通过"= 不可能失败的量)。
    修法: 负向对照里把 bucket_proj.bias **随机化**, 逼 startb 到处跑, 此时
    bucket_step=0 的动态步长必然改变窗口 ⇒ 检查必须失败。
    """
    from model import RCfg, EngramRuleLM

    def run(bs_val, shuffle_bias=False):
        c = RCfg(**dict(cfg_dict, bucket_step=bs_val))
        m = EngramRuleLM(c).to(device)
        m.load_state_dict(state)
        if shuffle_bias:
            g = torch.Generator(device="cpu").manual_seed(1234)
            with torch.no_grad():
                for blk in m.blocks:
                    blk.attn.bucket_proj.bias.copy_(
                        torch.randn(c.n_buckets, generator=g).to(device))
        m.eval()
        grab = {}

        def hook(_mod, _inp, out):
            grab["h"] = out.detach()
        hd = m.norm.register_forward_hook(hook)
        try:
            torch.manual_seed(7)
            x = torch.randint(0, c.vocab, (1, L_full), device=device)
            m(x)
            h_full = grab["h"][:, :L_cut].clone()
            m(x[:, :L_cut])
            h_cut = grab["h"].clone()
        finally:
            hd.remove()
        return float((h_full - h_cut).abs().max()), float(h_full.abs().max())

    pos_bs = cfg_dict.get("bucket_step", 0)
    d_trained, mag = run(pos_bs, False)            # 出厂权重原样
    d_pos, _ = run(pos_bs, True)                   # 窗口起点会变 + 步长为常量
    d_neg, _ = run(0, True)                        # 窗口起点会变 + 步长随 L 变
    assert pos_bs > 0, "v11 的 bucket_step 必须是正整数常量, 读到 %r" % pos_bs
    return dict(bucket_step=pos_bs, diff_trained=d_trained, diff_const_step=d_pos,
                diff_zero_step=d_neg, magnitude=mag,
                pos_pass=d_pos < PREFIX_TOL, neg_pass=d_neg > 10 * PREFIX_TOL)


# ----------------------------------------------------------------- main

@torch.no_grad()
def sync_cost_us(model, x, reps=200):
    """一次 device→host 同步在本机的真实代价, 以及一次 kernel 提交的代价。
    用来把 [B] 数出来的同步次数换算成毫秒 —— 这是 P4 能不能解释 [A] 的关键换算。"""
    small = torch.zeros((), device=DEV)
    for _ in range(20):
        float(small.sum())
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        float(small.sum())
    sync_us = (time.perf_counter() - t0) / reps * 1e6

    # 只提交不同步的基线
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        small.sum()
    launch_us = (time.perf_counter() - t0) / reps * 1e6
    if DEV == "cuda":
        torch.cuda.synchronize()
    return sync_us, launch_us


@torch.no_grad()
def op_profile(model, x, label, topn=8):
    """profiler 读数: 真实 kernel 数与 CUDA 忙时。**用来替代已作废的 P6。**"""
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU]
    if DEV == "cuda":
        acts.append(ProfilerActivity.CUDA)
    model(x)                                     # 预热
    with profile(activities=acts, record_shapes=False) as prof:
        model(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
    ka = prof.key_averages()
    total_launches = sum(e.count for e in ka)
    cuda_busy_us = sum(e.self_device_time_total for e in ka)
    ops = sorted(((e.count, e.key, e.self_cpu_time_total) for e in ka),
                 key=lambda t: -t[2])[:topn]
    return dict(label=label, launches=total_launches, cuda_busy_us=cuda_busy_us, ops=ops)


def _group_by_prompt(tok, rows):
    """按 prompt token 数分组(批生成只能同组同批)。"""
    from dialogue_small import format_prefix
    g = {}
    for i, r in enumerate(rows):
        plen = len(tok.encode(format_prefix(r), add_special_tokens=False).ids)
        g.setdefault(plen, []).append(i)
    return g


def main():
    from tokenizers import Tokenizer
    from model import RCfg, EngramRuleLM
    import dialogue_small as ds
    from bench_gpu import bench_matmul, PEAK_TFLOPS

    timeit_selfcheck()

    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    _, ev_rows, _ = ds.split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                    seed=29, return_meta=True)
    rows = ev_rows[:N_ROWS]
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    cfg = model.c
    print("DEV=%s torch=%s vocab=%d d=%d layers=%d heads=%d ffn_mult=%d window_max=%d "
          "n_buckets=%d bucket_step=%d ctx_limit=96"
          % (DEV, torch.__version__, cfg.vocab, cfg.d, cfg.n_layers, cfg.n_heads,
             cfg.ffn_mult, cfg.window_max, cfg.n_buckets, cfg.bucket_step))

    # ---- 门禁 (规则 22): 先测参照负载, 再看后面的速度数字能不能引用 ----
    gate_frac = float("nan")
    if DEV == "cuda":
        tf, _ = bench_matmul()
        gate_frac = tf / PEAK_TFLOPS
        print("\n[门禁] 参照 matmul %.2f TFLOPS = FP32 峰值的 %.0f%% => %s"
              % (tf, 100 * gate_frac,
                 "可用" if gate_frac > 0.3 else "★被抢占: 下面所有速度读数标注『受抢占影响』"))
        print("       显存 %.0f / %.0f MiB" % (torch.cuda.memory_allocated() / 2**20,
                                              torch.cuda.mem_get_info()[1] / 2**20))

    # ---- A 生成路径拆账 ----
    prof, outs = profile_generation(model, tok, rows)
    wall = prof["wall_s"]
    print("\n[A] %d 行 × ≤%d token 的真实 greedy_generate" % (N_ROWS, MAX_NEW))
    print("    wall %.2f s | %.3f s/行 | 生成 token %d (均值 %.1f/行)"
          % (wall, wall / N_ROWS, sum(prof["gen_tokens"]),
             sum(prof["gen_tokens"]) / max(1, len(prof["gen_tokens"]))))
    print("    host: forward %.2f s (%.0f%%) | _select_next_id %.2f s (%.0f%%) | 其它 %.2f s (%.0f%%)"
          % (prof["fwd_host_s"], 100 * prof["fwd_host_s"] / wall,
             prof["sel_host_s"], 100 * prof["sel_host_s"] / wall,
             prof["other_s"], 100 * prof["other_s"] / wall))
    if DEV == "cuda":
        print("    GPU event span %.2f s = wall 的 %.0f%%  ⇒ **这不是 GPU 忙时**, 见 §F" 
              % (prof["gpu_span_s"], 100 * prof["gpu_span_s"] / wall))
    else:
        print("    (无 GPU, 跳过 span 读数)")

    p1_ratio = prof["sel_host_s"] / max(prof["fwd_host_s"], 1e-9)
    print("    [P1] sel/fwd = %.2f => %s" % (p1_ratio, "P1 成立" if p1_ratio > 0.2
                                             else "P1 被否"))

    # ---- B 同步计数 ----
    xv = torch.randint(0, cfg.vocab, (1, 96), device=DEV)
    byte_map, special_ids = ds.bytelevel_token_bytes(tok)
    eos_id = tok.token_to_id("[EOS]")
    with torch.no_grad():
        model(xv)                       # 预热, 不计
        n_fwd, notes = count_syncs(lambda: model(xv))
        logits = model(xv)[0, -1]
        n_sel, _ = count_syncs(lambda: ds._select_next_id(
            logits, [], (0, 0x80, 0xBF), byte_map, special_ids, eos_id,
            0.0, 1.0, 0, True, None, b""))
    print("\n[B] 单次前向同步 %d 次 | 单次 _select_next_id 同步 %d 次 | %s"
          % (n_fwd, n_sel, "; ".join(notes)))
    print("    [P4] %s" % ("P4 成立: 每步 %d 次同步会把流水线排空" % (n_fwd + n_sel)
                          if n_fwd + n_sel > 0 else "P4 被否: 未测到同步"))

    # ---- C batch 曲线 ----
    bs_res = batch_scaling(model, 96, [1, 4, 16, 32, 64])
    print("\n[C] 前向 L=96 的 batch 曲线: " +
          "  ".join("B=%d %.3f ms" % (b, t) for b, t in bs_res.items()))
    sp32 = bs_res[1] / max(bs_res[32], 1e-9)
    print("    t(B=1)/t(B=32) = %.1f× => %s" % (sp32, "launch-bound(P3 成立)"
                                              if sp32 < 8 else "★P3 被否: 已经吃满算力"))

    # ---- D head 占比 ----
    hs = head_share(model, 96)
    print("\n[D] head 占比: 完整前向 %.3f ms | 去掉 head %.3f ms | 实测占比 %.0f%% | "
          "解析 head/(attn+ffn) = %.1f×"
          % (hs["full_ms"], hs["nohead_ms"], 100 * hs["measured_share"],
             hs["analytic_head_over_rest"]))
    print("    [P2] %s" % ("P2 成立" if hs["measured_share"] > 0.4 else "P2 被否(<40%)"))

    # ---- E KV cache 可行性 ----
    pi = prefix_invariance(ck["cfg"], ck["state_dict"])
    print("\n[E] 前缀不变性(bucket_step=%d), 隐状态最大差 %.3e (量级 %.2f)"
          % (pi["bucket_step"], pi["diff_trained"], pi["magnitude"]))
    print("    正向: 步长为常量 %d + 起点会变 => 差 %.3e => %s"
          % (pi["bucket_step"], pi["diff_const_step"], "通过" if pi["pos_pass"] else "不通过"))
    print("    负向: 步长随 L 变(0) + 起点会变 => 差 %.3e => %s"
          % (pi["diff_zero_step"],
             "如预期失败(检查有效)" if pi["neg_pass"] else "★如预期通过 ⇒ 检查本身无效"))
    print("    [P5] %s" % ("成立: KV cache 可做逐位等价" if (pi["pos_pass"] and pi["neg_pass"])
                          else "★不成立"))
    assert pi["neg_pass"], "负向对照没失败 ⇒ 前缀不变性检查不是判据, P5 不能引用"

    # ---- F 同步代价 + kernel 真实忙时 (替代已作废的 P6) ----
    sc_us, launch_us = sync_cost_us(model, xv)
    print("\n[F] 一次 device→host 同步 %.1f µs | 一次 kernel 提交 %.1f µs"
          % (sc_us, launch_us))
    print("    [B] 的 %d 次同步 × %.1f µs = **%.2f ms/步** (占 [A] 每步 %.1f ms 的 %.0f%%)"
          % (n_fwd + n_sel, sc_us, (n_fwd + n_sel) * sc_us / 1e3,
             (prof["wall_s"] / max(prof["steps"], 1)) * 1e3,
             100 * ((n_fwd + n_sel) * sc_us / 1e3)
             / max((prof["wall_s"] / max(prof["steps"], 1)) * 1e3, 1e-9)))
    pf = op_profile(model, xv, "forward")
    fwd_wall_us = bs_res[1] * 1e3                 # [C] 里 B=1 的单次前向耗时
    print("    单次前向: 算子条目 %d | CUDA 忙时 %.1f µs (对比 wall %.0f µs ⇒ GPU 忙时占 %.0f%%)"
          % (pf["launches"], pf["cuda_busy_us"], fwd_wall_us,
             100 * pf["cuda_busy_us"] / max(fwd_wall_us, 1e-9)))
    print("    host 侧最贵的算子: " + " | ".join(
        "%s×%d(%.0fµs)" % (k, c, u) for c, k, u in pf["ops"]))

    # ---- G 批生成的可批量 (决定要不要上批生成) ----
    #  P7: eval 行的 prompt token 数**高度集中**, 存在足够大的等长组 ⇒ 批生成可行。
    #      同组等长 ⇒ pos=arange(L) 对每一行都成立 ⇒ **模型一行都不用改**
    #      (不用左填充, 不用把 pos 打通; 已完成的行为空位用 filler 补齐长度即可)。
    #      **否证**: 落在 ≥4 行一组的行数不足 60% ⇒ P7 被否。
    from collections import Counter as _C
    _, ev_all, _ = ds.split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                    seed=29, return_meta=True)
    plens = [len(tok.encode(ds.format_prefix(r), add_special_tokens=False).ids)
             for r in ev_all]
    hist = _C(plens)
    groups = sorted(hist.values(), reverse=True)
    cover = sum(g for g in groups if g >= 4)
    print("\n[G] eval %d 行的 prompt 长度: 种类 %d, 组大小(前8) %s"
          % (len(plens), len(hist), groups[:8]))
    print("    落在 ≥4 行一组里的行数 %d/%d (%.0f%%) ⇒ P7 %s"
          % (cover, len(plens), 100 * cover / len(plens),
             "成立" if cover >= 0.6 * len(plens) else "被否"))
    if groups:
        g0 = groups[0]
        near = min(bs_res, key=lambda b: abs(b - g0))
        t1, tg = bs_res[1], bs_res[near]
        print("    实测前向 B=1 %.2f ms vs B=%d %.2f ms ⇒ 每行前向成本降到 %.1f%%"
              % (t1, near, tg, 100 * tg / max(near * t1, 1e-9)))

    # ---- H 端到端: 逐行 vs 等长批 (真实 eval 行, 与 §50 MBR 同口径) ----
    sub = ev_all                       # §50 MBR 的真实工作负载就是这 204 行
    groups = _group_by_prompt(tok, sub)
    gsizes = sorted((len(v) for v in groups.values()), reverse=True)
    over96 = sum(len(groups[k]) for k in groups if k > 96)
    ctx = ck["cfg"].get("pos_abs_max", 512)
    print("\n[H] %d 行真实 eval, evaluate_generations 端到端 (max_new_tokens=%d, context_limit=%d)"
          % (len(sub), MAX_NEW, ctx))
    print("    prompt>96  token 的行数 %d (默认 context_limit=96 的老口径装不下 ⇒ "
          "probe_rerank_mbr.py 在这些行上会直接抛错)" % over96)
    print("    等长组大小(前10) %s | 中位 %d | 组数 %d" % (gsizes[:10], gsizes[len(gsizes) // 2],
                                                    len(groups)))
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    a_sum, a_out = ds.evaluate_generations(model, tok, sub, MAX_NEW, DEV,
                                           context_limit=ctx, batch_min_size=0)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t_row = time.perf_counter() - t0
    t0 = time.perf_counter()
    b_sum, b_out = ds.evaluate_generations(model, tok, sub, MAX_NEW, DEV,
                                           context_limit=ctx, batch_min_size=3)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t_bat = time.perf_counter() - t0
    same = sum(1 for x, y in zip(a_out, b_out) if x["generated"] == y["generated"])
    print("    逐行 %.2f s (%.3f s/行) | 等长批 %.2f s (%.3f s/行) | **%.2f×** | 生成文本一致 %d/%d"
          % (t_row, t_row / len(sub), t_bat, t_bat / len(sub),
             t_row / max(t_bat, 1e-9), same, len(sub)))
    print("    EM 逐行 %.4f vs 批 %.4f (必须相同)" % (a_sum["exact_match"], b_sum["exact_match"]))
    assert same == len(sub), "批/逐行生成文本不一致 —— 速度读数没有意义, 先修等价性"
    assert a_sum["exact_match"] == b_sum["exact_match"], "批/逐行 EM 不一致"

    # ---- I 总账: 改动前(冻结参照) vs 改动后, **同一负载同一口径** ----
    # 用 _ref_preopt 的三段冻结实现当"改动前": 它就是优化前的逐字代码, 所以这个 A/B
    # 测的是同一台机器上同一份权重的前后差, 而不是"另一个版本"。
    import _ref_preopt as _rp
    from model import EngramMemory, RuleAttention
    saved = (EngramMemory.read, RuleAttention.forward)
    EngramMemory.read = _rp.ref_engram_read
    RuleAttention.forward = _rp.ref_rule_attention_forward
    try:
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        o_sum, o_out = ds.evaluate_generations(model, tok, sub, MAX_NEW, DEV,
                                              context_limit=ctx, batch_min_size=0)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t_old = time.perf_counter() - t0
    finally:
        EngramMemory.read, RuleAttention.forward = saved
    osame = sum(1 for x, y in zip(o_out, a_out) if x["generated"] == y["generated"])
    print("\n[I] 总账 (204 行 / max_new_tokens=%d / context_limit=%d, 同一份 v11 权重)" % (MAX_NEW, ctx))
    print("    改动前(冻结参照) %.2f s (%.3f s/行)" % (t_old, t_old / len(sub)))
    print("    改动后逐行         %.2f s (%.3f s/行)  ⇒ %.2f×"
          % (t_row, t_row / len(sub), t_old / max(t_row, 1e-9)))
    print("    改动后+等长批      %.2f s (%.3f s/行)  ⇒ **%.2f×**"
          % (t_bat, t_bat / len(sub), t_old / max(t_bat, 1e-9)))
    print("    生成文本与改动前一致 %d/%d | EM %.4f vs %.4f (必须相同)"
          % (osame, len(sub), o_sum["exact_match"], a_sum["exact_match"]))
    assert osame == len(sub), "优化改变了生成结果 —— 速度对比无效"
    assert o_sum["exact_match"] == a_sum["exact_match"] == b_sum["exact_match"]

    print("\n门禁 gate_frac=%.3f (引用速度比值前必须 >=0.3)" % gate_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
