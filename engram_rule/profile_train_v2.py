"""★profile v2 的训练步：找出**当前**最大的一块可省算力。

**为什么不直接照 §52.7 的推测做**：§52 的 profile 基准是 **N=738 / bs=256 / SEQ=512**
的小语料；现在 v2 是 **N=124,593**。行长分布、批内最大长度、micro_batch 切分都变了
⇒ 各算子占比会变。先重测再决定优化哪一块（规则 20：性能结论必须给 profile 读数）。

**口径**：与 `train_steps()` 逐字相同的前向/loss（同一 trim 逻辑、同一 answer_loss_sums），
只在外面包 `torch.profiler`。**不训练**，跑 20 步采样。

**踩过的坑（§52.1）**：
- `key_averages()` 里 **aten 算子层与它底下的 CUDA kernel 层各记一次 device time**，
  全加会重复计（我因此测出 194% / 754%，物理不可能）。
  ⇒ 只累加**非 `aten::`** 条目，并与 aten 层单独统计**互证**（两者应相等）。
- CUDA event 的 span ≠ GPU 忙时（覆盖 host 排队等待）。
  ⇒ 用 profiler 的 self CUDA time。
"""
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 20


def main():
    if DEV != "cuda":
        raise SystemExit("profile 需要 CUDA")
    import exp_scaling_grid_v2 as e
    e.load_all()   # 填 e._CK / e._TOK / e._TOK_META

    rows = e.load_rows()
    tr, pr = e.split_rows(rows)
    e.assert_no_leak(tr, pr)
    tok = e._TOK

    d = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 124593
    print("profile: N=%d d=%d，%d 步" % (n, d, STEPS), flush=True)

    import model as M
    # ★用 e._CK（exp_scaling_grid_v2 已加载好的 v11 dict），不要自己再 torch.load
    #   —— v11 在这个文件里是 dict 不是路径，直接 torch.load(dict) 会报
    #   "'dict' object has no attribute 'seek'"
    ck = e._CK
    cfg = dict(ck["cfg"]); cfg["d"] = d; cfg["n_heads"] = d // 16
    torch.manual_seed(11)
    model = M.EngramRuleLM(M.RCfg(**cfg)).to(DEV)
    x, s, l = e.DS.encode_examples(tr[:n], tok, 512)
    e.DS.build_engram_from_examples(model, x, l)

    # 行长分布（决定截断收益）
    lens = l.tolist()
    lens_sorted = sorted(lens)
    print("  行长 min/中位/p90/max = %d / %d / %d / %d"
          % (lens_sorted[0], lens_sorted[len(lens_sorted) // 2],
             lens_sorted[int(len(lens_sorted) * 0.9)], lens_sorted[-1]))
    print("  全量填充率 = %.1f%%（张量 512）"
          % (100.0 * (1.0 - sum(lens) / (len(lens) * 512))))
    # 批内真实最大长度（micro_batch=64 切分后）
    g = torch.Generator().manual_seed(11)
    idx = torch.randint(0, x.shape[0], (256,), generator=g)
    mb_max = []
    for i in idx.split(64):
        mb_max.append(int(l[i].max()))
    mb_sorted = sorted(mb_max)
    print("  批内最大长度 中位/p90/max = %d / %d / %d"
          % (mb_sorted[len(mb_sorted) // 2], mb_sorted[int(len(mb_sorted) * 0.9)],
             mb_sorted[-1]))

    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    model.train()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for step in range(STEPS):
            i = idx.split(64)[step % len(idx.split(64))]
            bx_full = x[i].to(DEV); sb = s[i].to(DEV); lb = l[i].to(DEV)
            bx = e.DS._trim_length(bx_full, lb)
            pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            tl, _fl, _na, _nf = e.DS.answer_loss_sums(logits, bx[:, 1:], valid, sb)
            (tl / (lb - sb).sum().clamp_min(1).to(DEV)).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            if DEV == "cuda":
                torch.cuda.synchronize()

    ka = prof.key_averages()
    # ★只累加非 aten:: 条目（避免算子层与 kernel 层重复计）
    k_us = sum(e.self_device_time_total for e in ka
               if not e.key.startswith("aten::"))
    a_us = sum(e.self_device_time_total for e in ka
               if e.key.startswith("aten::"))
    print()
    print("kernel 忙时合计 = %d µs   aten 层 = %d µs   (应相等, 互证)"
          % (k_us, a_us))
    print("每步 kernel 忙时 = %.1f ms" % (k_us / STEPS / 1000.0))
    print()
    rows_out = []
    for e in sorted(ka, key=lambda z: -z.self_device_time_total)[:14]:
        if e.self_device_time_total <= 0:
            continue
        rows_out.append((e.key, e.self_device_time_total / STEPS / 1000.0,
                         e.count // max(STEPS, 1)))
        print("  %-46s %8.2f ms/步  %5.1f%%  调用 %d/步"
              % (e.key[:46], e.self_device_time_total / STEPS / 1000.0,
                 100.0 * e.self_device_time_total / k_us,
                 e.count // max(STEPS, 1)))
    with open(os.path.join(HERE, "profile_v2_n%d_d%d.json" % (n, d)),
              "w", encoding="utf-8") as f:
        json.dump(dict(n=n, d=d, steps=STEPS, kernel_us_per_step=k_us / STEPS,
                       aten_us_per_step=a_us / STEPS,
                       rows=[dict(op=k, ms=m, calls=c) for k, m, c in rows_out]),
                  f, ensure_ascii=False, indent=1)
    print("\n已写出 profile_v2_n%d_d%d.json" % (n, d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
