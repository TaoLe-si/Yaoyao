"""§68.7：用 profiler 数**同步次数**与 **kernel 忙时**，逐项判决每个修复点。

**为什么数同步次数而不是只看墙钟**：同步的代价 ≈ 次数 × 49.5µs（§51 实测），
而墙钟混了机器负载漂移（本机 GPU 空闲利用率仍 99%）。
数 `cudaStreamSynchronize` 调用数是**确定性**指标，不受负载影响。

**做法**：分别加载 `dist/model.py`（改前）与 `model.py`（改后），
同一批输入、同一 seed，各跑 20 次前向，用 profiler 统计：
  · `cudaStreamSynchronize` 调用次数/前向
  · `aten::` 非 kernel 层的 self CUDA time（kernel 忙时）
  · host 侧 wall time

**负向对照**：`_check_pos=True` 恢复断言 ⇒ 同步次数必须**回升**，
否则说明 profiler 没在数这个东西（判据无效）。
"""
import importlib.util
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_FWD = 20


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@torch.no_grad()
def measure(mod, e, x, l, ck, d, tag):
    torch.manual_seed(11)
    m = mod.EngramRuleLM(mod.RCfg(**{**ck["cfg"], "d": d,
                                      "n_heads": d // 16})).to(DEV)
    e.DS.build_engram_from_examples(m, x, l)
    m.eval()
    xb = x[:4, :180].to(DEV)

    for _ in range(3):                       # 预热
        m(xb)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.time()
        for _ in range(N_FWD):
            m(xb)
        torch.cuda.synchronize()
        wall = (time.time() - t0) / N_FWD

    ka = prof.key_averages()
    sync = sum(ev.count for ev in ka if "StreamSynchronize" in ev.key
               or "cudaStreamSynchronize" in ev.key)
    kernel_us = sum(ev.self_device_time_total for ev in ka
                    if not ev.key.startswith("aten::"))
    # 数 nonzero（布尔掩码索引的特征）
    nz = sum(ev.count for ev in ka if "nonzero" in ev.key)
    del m
    torch.cuda.empty_cache()
    return dict(tag=tag, sync_per_fwd=sync / N_FWD, kernel_ms=kernel_us / N_FWD / 1000.0,
                wall_ms=wall * 1000.0, nonzero_per_fwd=nz / N_FWD)


def main():
    import exp_scaling_grid_v2 as e
    e.load_all()
    rows = e.load_rows()
    tr, he = e.split_rows(rows)
    e.assert_no_leak(tr, he)
    x, s, l = e.encode_cached(tr[:2000], e._TOK, 512)
    ck = e._CK
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 64

    new = load(os.path.join(HERE, "model.py"), "m_new")
    old = load(os.path.join(HERE, "dist", "model.py"), "m_old")

    print("=" * 74)
    print("同步次数 / kernel 忙时：改前(dist) vs 改后(main)，d=%d，%d 次前向"
          % (d, N_FWD))
    print("=" * 74)
    a = measure(old, e, x, l, ck, d, "改前(dist)")
    b = measure(new, e, x, l, ck, d, "改后(main)")
    for r in (a, b):
        print("  %-12s 同步 %.2f 次/前向   nonzero %.2f 次   kernel %.2f ms   wall %.2f ms"
              % (r["tag"], r["sync_per_fwd"], r["nonzero_per_fwd"],
                 r["kernel_ms"], r["wall_ms"]))
    print()
    print("  ⇒ 同步次数  %.2f → %.2f（省 %.1f%%）"
          % (a["sync_per_fwd"], b["sync_per_fwd"],
             100 * (1 - b["sync_per_fwd"] / max(a["sync_per_fwd"], 1e-9))))
    print("  ⇒ kernel 忙时 %.2f → %.2f ms（%.3fx）"
          % (a["kernel_ms"], b["kernel_ms"],
             a["kernel_ms"] / max(b["kernel_ms"], 1e-9)))
    print("  ⇒ wall      %.2f → %.2f ms（%.3fx）"
          % (a["wall_ms"], b["wall_ms"],
             a["wall_ms"] / max(b["wall_ms"], 1e-9)))

    print()
    print("=" * 74)
    print("负向对照：_check_pos=True 恢复 rope 断言 ⇒ 同步次数必须回升")
    print("=" * 74)
    orig = new.rope.__defaults__
    new.rope.__defaults__ = (10000.0, True)      # _check_pos=True
    c = measure(new, e, x, l, ck, d, "恢复断言")
    new.rope.__defaults__ = orig
    print("  %-12s 同步 %.2f 次/前向" % (c["tag"], c["sync_per_fwd"]))
    back = c["sync_per_fwd"] > b["sync_per_fwd"]
    print("  [%s] 恢复断言后同步次数 %s（%.2f vs %.2f）"
          % ("PASS" if back else "FAIL",
             "回升" if back else "★没变 ⇒ 判据无效",
             c["sync_per_fwd"], b["sync_per_fwd"]))
    return 0 if back else 1


if __name__ == "__main__":
    sys.exit(main())
