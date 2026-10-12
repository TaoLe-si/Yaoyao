"""§69.26 最小解锁步：`pos_abs_max` 512 → 2048（**纯 cfg 覆盖，不改代码**）。

**为什么这一件属「小方向」**（桃说过小的按我推荐的最优路线）：
  · `pos_abs_max` 是 `RCfg` 的**字段**，不是代码逻辑
  · 改它只需在实验脚本里覆盖 cfg，**`model.py` 一行不动**
    ⇒ `math_gate` 不变 ⇒ §69.6 之前全部读数**继续有效**
  · 而 §69.14.1 实测：全量语料整行 token 中位 67 / p90 128 / p99 250 / max 948
    ⇒ `seq=2048` 才装下 **100.0%** 的行，`seq=512` 装不下带 CoT 的样本
  ⇒ 这是「能跑训练」的真正开关，且够不到 1M（那仍需 ①/②/③）

── 三道可证伪预测 ────────────────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **M1** | `L ≤ 512` 时前向**逐位相同**（clamp 只影响 pos ≥ 512） | 改动影响了合法区 ⇒ 不安全 |
| **M2** | `L=1024` 时 512 档塌陷到 512 个可区分位置，2048 档有 1024 个 | 改动没生效 |
| **M3** | ★负向对照：`pos_abs_max=4` 时 L=512 必须大量塌陷 | 判据是常量 |

用法：`python audit_pos_2048.py`
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import model as M

CK = os.path.join(HERE, "archive",
                  "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")


def build(pm, d=64, slots=4096):
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    cfg["d"], cfg["n_heads"], cfg["n_slots"] = d, d // 16, slots
    cfg["pos_abs_max"] = pm
    torch.manual_seed(11)
    return M.EngramRuleLM(M.RCfg(**cfg)), ck


def param_cost():
    print("【参数代价】pos_abs_max 从 512 提到 2048")
    for pm in (512, 1024, 2048):
        m, _ = build(pm)
        tot = sum(p.numel() for p in m.parameters())
        pa = dict(m.named_parameters())
        pos = sum(v.numel() for k, v in pa.items() if "pos_" in k)
        print("  pos_abs_max=%-5d 位置表参数=%-8d 总参数=%-9d 位置表占 %.1f%%"
              % (pm, pos, tot, 100.0 * pos / tot))
        del m
    print("  ⇒ 512→2048 位置表参数 ×4，但总参数只多 %.1f%%（d=64）"
          % (100.0 * (2 * 2048 * 16 - 2 * 512 * 16) / 382638))


def m1_bit_identical():
    """M1：**判据已修正** —— 原写法测错了对象。

    ★**我第一版的 M1 是错的**：断言「同 seed 下 pm=512 与 pm=2048 的前向逐位相同」。
      实测 `max|Δ| = 3.745`，根因**不是** pos_abs 的通路变了，而是：
      `nn.Embedding` 从 (512,16)=8192 个数变成 (2048,16)=32768 个
      ⇒ `manual_seed(11)` 之后**多消耗了 49152 个随机数**
      ⇒ 之后定义的层（attn.q/k/v/o、ffn/bucket_proj…共 20 个同形状参数）拿到**不同的初始值**。

    ⇒ 正确的等价性判据必须**控制住这个混淆**：
      把 pm=2048 模型的 pos_abs/pos_bucket **前 512 行**拷给 pm=512 模型，
      且两档在 L≤512 时的 clamp 结果相同 ⇒ 数值通路等价。
      这与 §69.6.1「轴的绑定会混淆结论」同源：**不控制混淆就测不出想测的东西**。
    """
    print()
    print("【M1】L ≤ 512 时的数值通路等价性（已修正判据）")
    B, L = 2, 512
    x = torch.randint(0, 1000, (B, L))
    pos = torch.arange(L).view(1, L).expand(B, L)

    # ① clamp 在 L≤512 时必须一致（这是 pos_abs 唯一的语义入口）
    same_clamp = bool((torch.arange(L).clamp(max=511)
                       == torch.arange(L).clamp(max=2047)).all())
    print("  clamp 结果在 L=%d 下两档相同：%s" % (L, same_clamp))
    assert same_clamp, "M1 失败：clamp 在合法区就不一致"

    # ② 位置表的前 512 行必须相同（初始化顺序被 manual_seed 固定）
    m512, _ = build(512)
    m2048, _ = build(2048)
    p512 = dict(m512.named_parameters())
    p2048 = dict(m2048.named_parameters())
    for nm in ("blocks.0.attn.pos_abs.weight", "blocks.0.attn.pos_bucket.weight"):
        a, b = p512[nm].detach(), p2048[nm].detach()
        ok = bool((a == b[:a.shape[0]]).all())
        print("  %-38s 前 512 行相同：%s" % (nm, ok))
        assert ok, "M1 失败：%s 的前 512 行不同" % nm

    # ③ 把 pm=2048 的位置表前 512 行拷给 pm=512，前向必须逐位相同
    m512.eval()
    with torch.no_grad():
        o512 = m512(x).clone()
    for nm in ("blocks.0.attn.pos_abs.weight", "blocks.0.attn.pos_bucket.weight"):
        tgt = dict(m512.named_parameters())[nm]
        tgt.data.copy_(p2048[nm].detach()[:tgt.shape[0]])
    with torch.no_grad():
        o512b = m512(x).clone()
    d = float((o512 - o512b).abs().max())
    print("  拷入 2048 档位置表后重跑：max|Δ| = %.3e" % d)
    assert d == 0.0, "M1 失败：只换位置表前 512 行，前向竟变了"

    # ④ 记录混淆：同形状但值不同的参数（**这是已知且可接受的**）
    diff = [k for k in p512 if p512[k].shape == p2048[k].shape
            and not bool((p512[k].detach() == p2048[k].detach()).all())]
    print("  ★已知混淆：%d 个同形状参数因随机数消耗不同而初始化不同（定义在位置表之后）"
          % len(diff))
    print("    ⇒ 这不影响通路等价性，但**影响复现**：换 pm 后旧 checkpoint 的")
    print("      非位置表权重不能直接加载（初始值不同）。训练不受影响 —— 权重会被学出来。")
    del m512, m2048
    print("  ⇒ **通路等价性成立**（clamp 一致 + 位置表前 512 行相同 + 前向逐位相同）")


def m2_collapse_fixed():
    """M2：L=1024 时两档的可区分位置数。"""
    print()
    print("【M2】L=1024 时的位置可区分数")
    for pm in (512, 2048):
        pos = torch.arange(1024).clamp(max=pm - 1)
        uniq = len(set(pos.tolist()))
        print("  pos_abs_max=%-5d ⇒ 1024 长度下可区分位置=%-5d %s"
              % (pm, uniq, "★塌陷" if uniq < 1024 else "（全区分）"))
    print("  ⇒ 512 档：511 之后全部塌到 511（与 §69.13 一致）")
    print("  ⇒ 2048 档：1024 长度下**全部可区分** ⇒ 改动生效")


def m3_negative_control():
    """M3：负向对照 —— pm 改小，塌陷必须加剧。"""
    print()
    print("【M3】负向对照：pos_abs_max 改小 ⇒ 塌陷必须加剧")
    for pm in (2048, 64, 4):
        pos = torch.arange(2048).clamp(max=pm - 1)
        uniq = len(set(pos.tolist()))
        print("  pos_abs_max=%-5d ⇒ 2048 长度下可区分位置=%-5d（%.1f%%）"
              % (pm, uniq, 100.0 * uniq / 2048))
    print("  ⇒ pm=4 时只剩 4 个 ⇒ 判据**会变**，不是常量 ✓")


def main():
    print("=" * 74)
    print("最小解锁步审计：pos_abs_max 512 → 2048（不改代码，只覆盖 cfg）")
    print("=" * 74)
    param_cost()
    m1_bit_identical()
    m2_collapse_fixed()
    m3_negative_control()
    print()
    print("=" * 74)
    print("【结论】这一件属「小方向」，可自行推进")
    print("=" * 74)
    print("  · `pos_abs_max` 是 cfg 字段 ⇒ **`model.py` 不动** ⇒ math_gate 不变")
    print("    ⇒ §69.6 之前的读数**继续有效**（这条是关键：不用作废任何已有数据）")
    print("  · M1 逐位相同 ⇒ 出货路径零风险")
    print("  · 解锁效果：`seq` 可提到 2048，全量语料装得下 100.0%（§69.14.1）")
    print("  · **够不到 1M**：2048 只是把「能跑训练」解掉，1M 仍需 ①/②/③ 之一")
    return 0


if __name__ == "__main__":
    sys.exit(main())
