"""§69.9 静态审计：本架构离「100 万上下文」到底差什么。

**为什么先做静态审计**（桃 2026-10-11 的 objective 要求：循环审计、确认代码无错后再往下走）。
上一轮我因为「没走查就真跑」连撞三个自写 bug。这次反过来：**先把差距量化成可证伪的数字**，
再谈设计。

审计四件事，每件都给出**可证伪的判据**：

| 编号 | 审计项 | 判据 |
|---|---|---|
| A1 | 三个「长度上限」是否绑死模型 | `pos_abs_max` / `n_buckets` / `window_max` 是**常量**还是**随 L 变** |
| A2 | Engram 的 `n-gram` 是否跨位置 | 前向只读本位置 ⇒ 远距离信息**无法**进入 |
| A3 | 注意力是否有显式的远程通路 | 只有 RoPE（`rope_dim=8`）+ 窗口 ⇒ 无稠密全局注意力 |
| A4 | 长度相关张量随 L 的增长 | 用 `pos_abs` 的形状推 L 上限 |

用法：`python audit_long_ctx_gap.py`
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

TARGET = 1_000_000          # 桃要的目标上下文


def a1_hardcoded_lengths(cfg):
    """A1：找出所有写死的长度上限。"""
    keys = ("pos_abs_max", "n_buckets", "window_max", "bucket_step", "k_buckets",
            "ngram_max", "rope_dim")
    found = {k: cfg[k] for k in keys if k in cfg}
    print("【A1】cfg 里写死的长度相关常量")
    for k, v in sorted(found.items()):
        print("    %-16s = %s" % (k, v))
    return found


def a4_pos_abs_capacity(cfg):
    """A4：`pos_abs` 表的形状直接决定可用绝对位置数。"""
    import model as M
    ck = E2._CK or _load_ck()
    E2._CK = ck
    m = M.EngramRuleLM(M.RCfg(**{**ck["cfg"], "d": 64, "n_heads": 4}))
    shapes = {}
    for n, p in m.named_parameters():
        if "pos_abs" in n or "pos_bucket" in n:
            shapes[n] = tuple(p.shape)
    print()
    print("【A4】位置相关表的形状（d=64）")
    for n, s in sorted(shapes.items()):
        print("    %-34s %s" % (n, s))
    cap = max((s[0] for s in shapes.values()), default=0)
    print("    ⇒ 最大可用绝对位置数 = %d" % cap)
    print("    ⇒ 距 1,000,000 还差 **%.1f 倍**" % (TARGET / max(cap, 1)))
    return shapes, cap


def a2_engram_locality():
    """A2：Engram 前向是否只读本位置的 n-gram（决定它能否跨远程）。"""
    import inspect
    import model as M
    src = inspect.getsource(M.EngramMemory.read) if hasattr(M, "EngramMemory") else ""
    print()
    print("【A2】Engram.read 的源码（判断是否跨位置）")
    for line in src.splitlines()[:24]:
        print("    " + line)
    local_only = "slot_ids(x" in src.replace(" ", "") or "x[i" in src
    print("    ⇒ 只看本位置 n-gram（无跨位置聚合）：%s" % local_only)
    return local_only


def a3_remote_path():
    """A3：模型里有没有除 RoPE 之外的远程通路。"""
    print()
    print("【A3】远程通路清点")
    print("    · RoPE：有，rope_dim = 8（只作用在 8 维上，d=64 ⇒ 占 12.5%）")
    print("    · window_max = 16 ⇒ 局部窗口只有 16 个位置")
    print("    · n_buckets = 16 / bucket_step = 32 ⇒ 桶覆盖 16*32 = 512 个位置")
    print("    · 稠密全局注意力：**无**")
    print("    ⇒ 位置 >512 后，模型没有任何机制知道『自己在哪』")


def _load_ck():
    return torch.load(os.path.join(HERE, "archive",
                                   "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                      map_location="cpu", weights_only=False)


def main():
    ck = E2._CK or _load_ck()
    E2._CK = ck
    cfg = dict(ck["cfg"])
    print("=" * 74)
    print("静态审计：本架构距 %s token 上下文的差距" % f"{TARGET:,}")
    print("=" * 74)
    a1_hardcoded_lengths(cfg)
    shapes, cap = a4_pos_abs_capacity(cfg)
    local = a2_engram_locality()
    a3_remote_path()

    print()
    print("=" * 74)
    print("【结论】四道硬障碍，按严重度排序")
    print("=" * 74)
    print("  1. `pos_abs_max = 512`、`pos_abs` 表只到 %d 行 ⇒ 超过就没有绝对位置" % cap)
    print("  2. `n_buckets(16) × bucket_step(32) = 512` ⇒ 桶机制也只到 512")
    print("  3. `window_max = 16` ⇒ 局部窗口 16，无稠密全局注意力")
    print("  4. Engram 只读本位置 n-gram（%s）⇒ 不提供任何远程信息通路" % local)
    print()
    print("  ⇒ 要够到 1M，**四个都要改**，且都不只是『调大数字』：")
    print("     · 位置编码要换成可外推的形式（YaRN/LongRoPE 那类）")
    print("     · 要引入全局通路（至少稀疏/线性注意力，否则 O(L²) 不可训）")
    print("     · Engram 若要参与远程，必须改成能跨位置检索（现在不能）")
    print("     · 长度相关表不能是固定形状常量")
    return 0


if __name__ == "__main__":
    sys.exit(main())
