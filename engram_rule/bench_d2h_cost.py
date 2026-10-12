"""§68.7 修复点的**直接**计时：量 `int(pos[0])` 这类 D2H 同步本身花了多少。

**为什么换尺子**：profiler 的 `cudaStreamSynchronize` 计数**量不到** `int(张量)`
（负向对照已证：恢复 `_check_pos=True` 后计数不变 ⇒ 判据无效）。
`int(张量)` 走 PyTorch 内部 `.item()`，不一定在 profiler 留 CUDA API 条目。

**新尺子**：直接把 `int(pos[0])` 这一步单独计时 —— 同一条语句跑 N 次取中位。
这是**可失败**的：把 `int(pos[0])` 换成常量 0，耗时必须塌下去。

**三个修复点分别量**：
  P1  rope 断言  `int(pos[0])` + `int(pos[-1])`  （8 次/前向：2 层 × 2 调 × 2 元素）
  P2  read 的闸 `bool(any_live.any())`            （1 次/前向）
  P3  模型层闸   `bool((~skip).any())`            （1 次/前向）
"""
import os
import statistics
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REP = 2000


def bench(fn, rep=REP, warm=50):
    for _ in range(warm):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(rep):
        fn()
    return (time.time() - t0) / rep * 1000.0     # ms/次


def main():
    if DEV != "cuda":
        raise SystemExit("需要 CUDA")
    L = 512
    pos = torch.arange(L, device=DEV)
    any_live = torch.ones(4, L, dtype=torch.bool, device=DEV)
    # skip 是**布尔**张量（`read` 的返回值就是 `~any_live`），不是随机浮点
    skip = torch.rand(4, L, device=DEV) < 0.5

    # 先灌点别的 kernel，让 sync 真的要排空队列（更接近真实训练）
    junk = torch.randn(256, 256, device=DEV)

    def busy():
        return junk.mm(junk).sum()

    cases = [
        ("P1a  int(pos[0])       [1 元素 D2H]",
         lambda: int(pos[0])),
        ("P1b  int(pos[0])+int(pos[-1]) [2 元素]",
         lambda: (int(pos[0]), int(pos[-1]))),
        ("P2   bool(any_live.any())       [(4,512)->scalar]",
         lambda: bool(any_live.any())),
        ("P3   bool((~skip).any())        [(4,512)->scalar]",
         lambda: bool((~skip).any())),
        ("对照 pos[0].item()              [同义写法]",
         lambda: pos[0].item()),
        ("对照 pos.cpu()  [整张搬回]",
         lambda: pos.cpu()),
        ("对照 常量比较 0==0               [不该有 GPU 成本]",
         lambda: 0 == 0),
    ]
    print("=" * 72)
    print("D2H 同步的直接成本（L=%d，各 %d 次取均值）" % (L, REP))
    print("=" * 72)
    res = {}
    for tag, fn in cases:
        fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        plain = bench(fn)
        # 有前置计算时再测一次（同步要排空队列）
        busy(); fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(REP):
            busy()
            fn()
        withbusy = (time.time() - t0) / REP * 1000.0 - bench(busy, rep=200, warm=20)
        res[tag] = (plain, withbusy)
        print("  %-42s 空转 %6.4f ms   队列非空 %6.4f ms"
              % (tag, plain, withbusy))

    print()
    print("=" * 72)
    print("换算到每前向（d=64，n_layers=2）")
    print("=" * 72)
    k1 = res["P1a  int(pos[0])       [1 元素 D2H]"][1]
    k1b = res["P1b  int(pos[0])+int(pos[-1]) [2 元素]"][1]
    k2 = res["P2   bool(any_live.any())       [(4,512)->scalar]"][1]
    k3 = res["P3   bool((~skip).any())        [(4,512)->scalar]"][1]
    n1 = (k1b * 4) * 2      # 2 次 int/rope × 4 次 rope 调用(2层×q,k) ... 见下
    # rope 每层调 2 次(q/k)，每调 int 2 次 ⇒ 每层 4 次，2 层 = 8 次
    n1 = k1b * 8
    n2 = k2 * 1
    n3 = k3 * 1
    print("  P1 rope 断言   8 次 × %.4f ms = %.4f ms/前向" % (k1b, n1))
    print("  P2 read 的闸   1 次 × %.4f ms = %.4f ms/前向" % (k2, n2))
    print("  P3 模型层闸    1 次 × %.4f ms = %.4f ms/前向" % (k3, n3))
    tot = n1 + n2 + n3
    print("  ⇒ 三处合计 %.4f ms/前向" % tot)
    print()
    print("★对照：§68.1 探针评测 + 单步训练里前向约占多少")
    print("  单步(4 微批 × 2 层前向) 0.29 s ⇒ 单次前向 ~0.29/4 ≈ 72 ms")
    print("  ⇒ 三处占单次前向的 %.2f%%" % (100 * tot / 72.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
