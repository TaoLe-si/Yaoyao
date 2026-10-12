"""§69.32 「加了通路就能到 1M」—— 把这句话拆成两问，分别算清。

── 桃的反驳 ────────────────────────────────────────────────────────
「你添加通路不就能到了。」

── 他说对了什么 ────────────────────────────────────────────────────
§69.29 的 W3 表里，32 槽 × r=2.0 的几何采样最大回看 = 2.68e+08，
**远超 1,000,000**。⇒ **可达距离这一维，加了确实能到**，不需要分层。
我上一轮说「1M 这个目标在结构上到不了」是错的：把「当前架构」当成了「改造后的架构」。

── 我漏了什么 ──────────────────────────────────────────────────────
「位置索引能取到 1M 之外」与「在本机装得下、跑得动一条 1M 的序列」是两件事。
后者卡在**显存**，而这条我上一轮完全没算。本脚本把两问分开算。

── 四条可证伪读数 ──────────────────────────────────────────────────
| 编号 | 读什么 | 判据 |
|---|---|---|
| **R1** | 加稀疏通路后，索引可达距离够不够 1M | 32 槽 r=2 → 2.68e8 ≥ 1e6 ⇒ **够** |
| **R2** | 出货 bs=256 / L=512 下，SwiGLU 中间张量多大 | 对照基线，用于外推 |
| **R3** | L=1,000,000 单条（B=1）前向要多少显存 | 与 8188 MiB 比 ⇒ **超多少** |
| **R4** | 要装下 1M 前向，本机最多能开几个头 / 多小的 Dg | 给出可行档 |

★**R2 的口径更正**：`design_sparse_attention.py` 用的是 d=64 / n_heads=4
  （那是 `train_cot_smoke.py` 的配置），而**出货基准**是 d=128 / n_heads=8
  （`arch_figure_facts.json`）。同一个 W，显存差 4 倍。本脚本两个口径都算，
  出货结论以 **d=128 / H=8** 为准。

用法：`python probe_1m_feasibility.py`（纯算术，不加载模型）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "feasibility_1m_facts.json")

VRAM_MIB = 8188                      # 本机实测总显存
TARGET = 1_000_000
MB = 1024 * 1024


def swiglu_bytes(B, L, W, H, Dg, dtype_bytes=4):
    """`gate` / `up` 的中间张量 (B, L, W, H, Dg) 的字节数。

    ★为什么只算这两个：`model.py:475-476` 里 `gate` 与 `up` 都是这个形状且都
      要物化（`local_norm=True` 时 `gate` 还要再做一次 abs+sum）。其余量要么
      是 (B,L,H,D) 小一档，要么被原地复用。
    """
    return B * L * W * H * Dg * dtype_bytes


def main():
    print("=" * 78)
    print("「加了通路就能到 1M」—— 拆成可达距离 / 显存两问分别算")
    print("=" * 78)

    # ── R1 可达距离 ────────────────────────────────────────────────
    print()
    print("【R1】可达距离：加稀疏通路够不够 1M")
    print("  几何采样 第 k 槽回看 step * r^k，最大回看 = step * r^(槽数-本地槽数)")
    for n in (16, 32):
        for r in (1.5, 2.0, 3.0):
            far = 32 * r ** (n - 8)
            print("    %2d 槽  r=%.1f  ->  最远 %12.4g   %s"
                  % (n, r, far, "够 1M" if far >= TARGET else "不够"))
    far32 = 32 * 2.0 ** (32 - 8)
    print("  ⇒ 32 槽 × r=2.0 最远 %.4g，是 1M 的 %.1f 倍 ⇒ **可达距离够，不需要分层**"
          % (far32, far32 / TARGET))
    print("  ★这一条我上轮说错了：我说「1M 的最后一程要靠分层」——那是拿 16 槽算的。")
    print("    槽数加到 32，单层就到 %.3g，分层不是必需的，只是省显存的一个选项。" % far32)

    # ── R2/R3 显存 ─────────────────────────────────────────────────
    print()
    print("【R2】出货配置的显存基线（d=128 / n_heads=8 / Dg=16）")
    rows = []
    for tag, d, H, Dg in (("出货 d=128 H=8", 128, 8, 16),
                          ("冒烟 d=64  H=4", 64, 4, 16)):
        W_cont = 16
        B, L = 256, 512
        b = swiglu_bytes(B, L, W_cont, H, Dg)
        rows.append(dict(case=tag, d=d, H=H, Dg=Dg, W=W_cont, B=B, L=L,
                         mb=round(b / MB, 1)))
        print("  %-16s B=%-4d L=%-5d W=%-3d  gate+up = %7.1f MB"
              % (tag, B, L, W_cont, b / MB))
    print("  ★这两个口径差 4 倍（d 与 H 同时翻倍）。design_sparse_attention.py 报的")
    print("    「512 MB 与现有同量级」是 **d=64** 口径；出货 d=128 是 **1024 MB**。")

    # ── R3 1M 单条前向 ─────────────────────────────────────────────
    print()
    print("【R3】L=1,000,000 单条前向（B=1）要多少显存")
    H, Dg = 8, 16
    cases = []
    for label, W in (("只跑现有连续窗口", 16),
                     ("现有 + 稀疏 16 槽（两条支路各算）", 16 + 16)):
        b = swiglu_bytes(1, TARGET, W, H, Dg)
        cases.append(dict(case=label, W=W, mb=round(b / MB, 1),
                          over=round(b / MB / (VRAM_MIB / 1024.0), 1)))
        print("  %-32s W=%-3d  gate+up = %8.1f GB  = 本机显存的 %.1f 倍"
              % (label, W, b / 1024.0 ** 3, b / MB / (VRAM_MIB / 1024.0)))
    print("  ★这**只是前向的中间张量**，还没算：")
    print("    · 反向传播要保存激活（实测通常再 2-3 倍）")
    print("    · emb 输出 (B,L,d) = 1e6 × 128 × 4 = 512 MB，logits (B,L,V) = 1e6×1024×4 = 4 GB")
    logits_gb = TARGET * 1024 * 4 / 1024.0 ** 3
    print("    · 单是 logits 就 %.1f GB，本机 %.2f GB ⇒ **光 logits 就装不下**"
          % (logits_gb, VRAM_MIB / 1024.0))
    print("  ⇒ **1M 单条前向在本机不可行**，与加不加通路无关。")

    # ── R4 可行档 ──────────────────────────────────────────────────
    print()
    print("【R4】本机装得下的最长单条序列（B=1，只算 gate+up，不含 logits 与激活）")
    print("  %8s %8s %10s" % ("L", "W", "gate+up"))
    for L in (2048, 8192, 32768, 65536, 131072, 1_000_000):
        for W in (16, 32):
            b = swiglu_bytes(1, L, W, H, Dg)
            mark = "" if b / MB < VRAM_MIB / 2 else "   > 半卡"
            print("  %8d %8d %8.1f MB%s" % (L, W, b / MB, mark))
    maxL = VRAM_MIB / 2.0 * MB // (W if False else 32 * H * Dg * 4)
    print("  ⇒ 只用半卡（%.0f MiB）给中间张量：W=32 时 L 上限约 %s"
          % (VRAM_MIB / 2.0, "{:,}".format(int(maxL))))
    print("    （不含 logits / emb / 激活 / 优化器 ⇒ 实际上限远低于此）")

    facts = dict(
        target=TARGET, vram_mib=VRAM_MIB,
        r1_far_with_32slots=far32,
        r1_reach_ok=bool(far32 >= TARGET),
        r2_baseline=rows,
        r3_one_forward=cases,
        logits_gb=round(logits_gb, 2),
        r4_max_L_at_half_vram=int(maxL),
        d=128, H=H, Dg=Dg,
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 78)
    print("【结论】把「加了通路就能到 1M」拆开答：")
    print("=" * 78)
    print("  **可达距离 —— 你是对的，加了就能到。** 32 槽 × r=2 单层就到 %.3g，"
          % (far32 / 1e9))
    print("     是 1M 的 %.0f 倍，不需要分层。我上轮说「要靠分层」是拿 16 槽算的，算错了。" % (far32 / TARGET))
    print()
    print("  **但显存是另一道独立的墙，这道我上轮完全没算：**")
    print("     · 1M 单条前向的 gate+up 中间张量就要 %.0f GB，本机 %.2f GB" %
          (cases[-1]["mb"] / 1024.0, VRAM_MIB / 1024.0))
    print("     · 单是 logits (1, 1e6, 1024) 就 %.1f GB，已经超整卡" % logits_gb)
    print("     · 反向还要存激活，实际再 2-3 倍")
    print()
    print("  ⇒ 正确的说法：**稀疏通路解决「看得见多远」；1M 还要解决「装不装得下」。**")
    print("     前者是架构改动（右半那张图），后者是系统工程，两件事互不替代。")
    print("     本机上，1M 只能做**推理/评测**（滑窗 + 分块），**训练**1M 不可行。")
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
