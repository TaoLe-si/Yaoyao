# -*- coding: utf-8 -*-
"""★§69.28 真正的「向后兼容逐位相同」判据 —— 用来给 MGATE 迁移背书。

── 为什么必须新写这个 ──────────────────────────────────────────────────
`check_think_loss.py` 的 L1 名字叫「不改动时行为逐位相同」，但它实际做的是
`r1 = f(...)` / `r2 = f(...)`，**两次都是新实现**，比的是自洽性（确定性），
**从来没有和旧实现对比过**。⇒ 它不能作为「改 `dist/dialogue_small.py`
不影响出货路径训练数值」的证据。这正是 AGENTS 规则 10 的同义反复：
一个已经报过的、其实没测到的"证据"。

── 这个脚本测什么 ─────────────────────────────────────────────────────
用 `ast` 把**两个文件里 `answer_loss_sums` 的源码原文**分别抽出来（不重打副本），
在隔离命名空间里 exec，然后喂同一批随机输入逐位对拍四个返回值。

| 编号 | 判据 | 方向 |
|---|---|---|
| **B1** | `think_start=None` 时，dist 版 ≡ 根目录版，四个返回值**逐位相同** | 必须完全相等 |
| **B2** | ★负向对照：给了 `think_start` 后**必须不等** | 须有差异 |

★B2 是 B1 的可失败性证明：如果连给 think 都算不出差别，
  那 B1 的「相等」可能只是这个比较根本不敏感（恒真），不能用。

用法：`python check_think_backcompat.py`
"""
import ast
import io
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

OLD_SRC = os.path.join(HERE, "dialogue_small.py")            # 旧实现（无 think 参数）
NEW_SRC = os.path.join(HERE, "dist", "dialogue_small.py")    # 新实现（有 think）


def extract_func(path, name):
    """从源码文件里取出 `def name(...)` 的**原文**（含缩进还原）。

    ★为什么要抽源码而不是 import：根目录 `dialogue_small.py` 第 20~21 行
      `import train` / `from train import ...`，而 `train.py` 已归档进 `archive/`，
      整模块 import 会失败（这正是 AGENTS 规则 7 那次踩的坑）。
      抽源码只依赖 `torch` 与 `F`，与被测函数之外的副作用完全解耦。
    """
    with io.open(path, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = src.splitlines()
            start = node.lineno - 1
            # 用节点的 end_lineno 精确取到函数结束（Python 3.8+）
            end = node.end_lineno
            return "\n".join(lines[start:end])
    raise RuntimeError("%s 里找不到 def %s" % (path, name))


def compile_isolated(src_text):
    ns = {"torch": torch, "F": F}
    exec(compile(src_text, "<extracted>", "exec"), ns)     # noqa: S102  (故意动态执行)
    return ns["answer_loss_sums"]


def main():
    old_src = extract_func(OLD_SRC, "answer_loss_sums")
    new_src = extract_func(NEW_SRC, "answer_loss_sums")
    print("=" * 78)
    print("B1 向后兼容：dist 版(think_start=None) vs 根目录版，逐位对拍")
    print("=" * 78)
    print("  旧实现（%s）签名行: %s"
          % (os.path.relpath(OLD_SRC, HERE), old_src.splitlines()[0].strip()))
    print("  新实现（%s）签名行: %s"
          % (os.path.relpath(NEW_SRC, HERE), new_src.splitlines()[0].strip()))

    old = compile_isolated(old_src)
    new = compile_isolated(new_src)
    import inspect
    print("  inspect 复核：旧 %s" % inspect.signature(old))
    print("  inspect 复核：新 %s" % inspect.signature(new))
    assert "think_start" not in inspect.signature(old).parameters, \
        "★旧实现里已经有 think_start 了 ⇒ 根目录那份不是'旧版'，对照无效"
    assert "think_start" in inspect.signature(new).parameters, \
        "★新实现没有 think_start ⇒ 取到的不是新版本"

    g = torch.Generator().manual_seed(2026)
    TRIALS = 40
    worst = [0.0, 0.0, 0.0, 0.0]      # 四个返回值的最大绝对差
    n_diff = [0, 0, 0, 0]
    # 形状覆盖：训练实况 (micro_batch=64, T=511) 起步，也带小形状与退化形状
    shapes = [(64, 511, 97), (1, 1, 5), (3, 8, 2), (8, 511, 97), (2, 127, 11)]

    for k in range(TRIALS):
        B, T, V = shapes[k % len(shapes)]
        logits = torch.randn(B, T, V, generator=g)
        target = torch.randint(0, V, (B, T), generator=g)
        valid = torch.rand(B, T + 1, generator=g) > 0.15      # 随机挖洞，贴近真实补齐
        starts = torch.randint(0, T + 1, (B,), generator=g)

        r_old = old(logits, target, valid, starts)
        r_new = new(logits, target, valid, starts)
        assert len(r_old) == len(r_new) == 4, "返回值个数变了：%d vs %d" % (len(r_old), len(r_new))
        for i in range(4):
            a, b = r_old[i], r_new[i]
            d = float((a.double() - b.double()).abs().max())
            worst[i] = max(worst[i], d)
            if not torch.equal(a, b):
                n_diff[i] += 1

    names = ["loss_sum", "first_sum", "answer_mask.sum()", "first_mask.sum()"]
    print()
    print("  %d 组随机输入（形状覆盖 %s）" % (TRIALS, sorted({s[0] for s in shapes})))
    for i in range(4):
        print("    %-20s max|Δ| = %.3e  逐位不等次数 = %d" % (names[i], worst[i], n_diff[i]))
    bad = [names[i] for i in range(4) if worst[i] != 0.0 or n_diff[i] != 0]
    assert not bad, "B1 失败：以下返回值不是逐位相同 —— %s" % bad
    print("  ⇒ 四个返回值全部 max|Δ| = 0.000e+00 ⇒ **B1 成立**")

    print()
    print("=" * 78)
    print("B2 负向对照：同一个比较，在「给了 think_start」时必须**不等**")
    print("=" * 78)
    B, T, V = 64, 511, 97
    logits = torch.randn(B, T, V, generator=g)
    target = torch.randint(0, V, (B, T), generator=g)
    valid = torch.rand(B, T + 1, generator=g) > 0.15
    starts = torch.randint(1, T + 1, (B,), generator=g)      # ≥1，否则 think_start 无法严格小于它
    think_start = torch.clamp(starts - 5, min=0)             # 严格小于 starts，合法
    assert bool((think_start < starts).all()), "★B2 前提失效：构造出的 think_start 越界"

    base = old(logits, target, valid, starts)
    with_think = new(logits, target, valid, starts, think_start=think_start, think_weight=0.6)
    d_think = float((base[0].double() - with_think[0].double()).abs().max())
    n_mask_old = int(base[2])
    n_mask_new = int(with_think[2])
    print("  loss_sum: 旧 %.4f → 带 think %.4f，max|Δ| = %.4e" % (float(base[0]), float(with_think[0]), d_think))
    print("  计入 loss 的 token 数: 旧 %d → 新 %d（+%d）" % (n_mask_old, n_mask_new, n_mask_new - n_mask_old))
    assert d_think > 0.0, "B2 失败：给了 think_start 竟算不出差别 ⇒ B1 的'相等'不可信"
    assert n_mask_new > n_mask_old, "B2 失败：think 段没进 mask"
    print("  ⇒ 比较本身是**敏感**的（B1 的相等不是因为它什么都测不出来）⇒ **B2 成立**")

    print()
    print("=" * 78)
    print("【结论】给 MGATE 迁移用的证据，现在是真的了：")
    print("  `dist/dialogue_small.py` 加 think 支持后，**出货路径**")
    print("  （`answer_loss_sums(logits, target, valid, starts)`，从不传 think_start）")
    print("  的四个返回值与改动前**逐位相同**，max|Δ| = 0.000e+00，跨 %d 组随机输入。" % TRIALS)
    print("  ⚠ 这**不等于**「旧 checkpoint 能直接 resume」—— 参数初始化随机数消耗变了，")
    print("    那是 §69.26 已经单独记过的事。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
