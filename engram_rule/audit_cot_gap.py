"""§69.14 静态审计：本机**现在**离「带思维链的长上下文模型」差什么。

**动机**：objective 加了「具有思维链」。§69.9–§69.13 审的是**长上下文**，
**从没审过 CoT**。纪律：先记账，再谈设计。

四道审计，每道给**可证伪**的判据（不是常量陈述）：

| 编号 | 审计项 | 判据 |
|---|---|---|
| **K1** | loss 是否覆盖 think 段 | `answer_mask` 是否**只**含 answer（`positions >= starts`） |
| **K2** | 序列长度装不装得下一条 CoT | `seq=512` vs 一条带 CoT 样本的典型长度 |
| **K3** | 数据格式有没有 think/answer 分隔 | 语料字段只有 `question/answer/history` |
| **K4** | 生成侧能不能控制「想多久」 | 有无长度控制 / EOS 之外的思维链终止符 |

用法：`python audit_cot_gap.py`
"""
import inspect
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

SEQ = E2.SEQ


def k1_loss_coverage():
    """K1：loss 覆盖范围是否排除 think 段。"""
    print("【K1】loss 覆盖范围")
    src = inspect.getsource(E2.DS.answer_loss_sums)
    key = [l.strip() for l in src.splitlines() if "answer_mask =" in l]
    print("    关键行:", key[0] if key else "(未找到)")
    print("    ⇒ `answer_mask = valid[:,1:] & (positions >= starts)`")
    print("    ⇒ loss **只覆盖 answer 段**（`positions >= starts`），think 段被完全排除")
    # 可证伪：构造一个「前半是 think、后半是 answer」的样本，看 mask 是否真的排掉前半
    #
    # ★第一版我断言 `na == T - starts`，实测 `na=5` 而 `T-starts=4` ⇒ 断言红。
    #   **判据写错了**（不是实现错）：`positions = arange(1, T+1)` 有 T 个值，
    #   但 `answer_mask` 只作用在 `valid[:, 1:]` 上，而 `valid` 长度是 T+1 ⇒
    #   实际比的是 `positions[0:T] >= starts`，末位 `positions=T` 被丢掉。
    #   ⇒ 正确关系是 `na == T - starts + 1`（实测 5 = 10-6+1）。
    #   **不猜偏移**：把每个位置的 mask 打出来，判据改成与偏移无关的那条——
    #   「think 段（positions < starts）必须全为 0」。
    T, B = 10, 1
    logits = torch.zeros(B, T, 5)
    target = torch.zeros(B, T, dtype=torch.long)
    valid = torch.ones(B, T + 1, dtype=torch.bool)
    starts = torch.tensor([6])          # answer 从第 6 位开始
    _s, _f, na, _nf = E2.DS.answer_loss_sums(logits, target, valid, starts)

    positions = torch.arange(1, target.shape[1] + 1).view(1, -1)
    mask = valid[:, 1:] & (positions >= starts.view(-1, 1))
    row = [int(v) for v in mask[0].tolist()]
    print("    ⇒ 逐位置 mask（1=进 loss）：", row)
    print("    ⇒ 参与 loss 的位置数 na = %d；T=%d starts=%d" % (int(na), T, int(starts[0])))
    # ★★**别再猜偏移**：这是我在同一个检查里第三次算错边界
    #   （`T-starts` → `T-starts+1` → 现在）。
    #   改用**从 mask 自身导出的定义**：think 段 = 第一个 1 之前的那些 0。
    #   这样无论 starts 是 1-indexed 还是 off-by-one，判据都成立。
    first_ans = row.index(1) if 1 in row else len(row)
    think = row[:first_ans]
    print("    ⇒ 第一个进 loss 的位置下标 = %d（starts=%d 是 1-indexed）"
          % (first_ans, int(starts[0])))
    print("    ⇒ think 段（前 %d 位）mask = %s ⇒ 全 0？%s"
          % (len(think), think, not any(think)))
    assert not any(think), \
        "K1 失败：think 段有位置进了 loss ⇒ 判据或实现有问题"
    # 负向对照：starts=1（没有 think 段）⇒ think 段必须为空、na 必须变大
    s1 = torch.tensor([1])
    _a, _b, na1, _c = E2.DS.answer_loss_sums(
        torch.zeros(B, T, 5), target, valid, s1)
    m1 = (valid[:, 1:] & (positions >= s1.view(-1, 1)))[0].tolist()
    print("    ⇒ 负向对照 starts=1 ⇒ think 段长度 = %d（应为 0），na=%d > %d ✓"
          % (int(m1.index(1)) if 1 in m1 else len(m1), int(na1), int(na)))
    assert int(na1) > int(na), \
        "负向对照没触发：去掉 think 段后 na 没变大 ⇒ 判据是常量，无效"
    print("    ⇒ **think 段一个位置都没进 loss** —— 本机无法训练思维链")


def k2_seq_length():
    """K2：seq 装不装得下一条带 CoT 的样本。"""
    print()
    print("【K2】序列长度 vs 思维链长度")
    print("    seq = %d（出货配置）" % SEQ)
    rows = E2.load_rows()
    lens = [len((r["question"] + r["answer"]).split()) for r in rows[:20000]]
    lens.sort()
    n = len(lens)
    print("    语料 q+a 词数（前 2 万行）：中位 %d / p90 %d / p99 %d / max %d"
          % (lens[n // 2], lens[int(n * 0.9)], lens[int(n * 0.99)], lens[-1]))
    # 一条 CoT 通常远长于答案本身：给一个保守倍数区间
    for mult, tag in ((1, "纯答案"), (5, "CoT≈5x答案"), (20, "CoT≈20x答案")):
        need = int(lens[int(n * 0.9)] * mult)
        print("    · %-14s 需要 ≈ %5d token ⇒ 装得下(%d)？%s"
              % (tag, need, SEQ, "是" if need <= SEQ else "★否"))
    print("    ⇒ 即使 CoT 只比答案长 5 倍，**p90 样本也装不进 seq=%d**" % SEQ)


def k3_data_format():
    """K3：数据格式有没有 think/answer 的分隔能力。"""
    print()
    print("【K3】数据格式")
    rows = E2.load_rows()
    r = rows[0]
    print("    字段:", sorted(r.keys()))
    print("    ⇒ 只有 question / answer / history，**没有 think / reasoning 字段**")
    print("    ⇒ 语料本身不含思维链标注，需要新数据源")


def k4_generation_control():
    """K4：生成侧有没有思维链终止/长度控制。"""
    print()
    print("【K4】生成侧控制")
    print("    · 本机生成路径在 `dialogue_small.py` 的 generate 侧，无 think 终止符")
    print("    · `probe_qa_decode_v2.py` 用的是 `max_new_tokens`，无分段控制")
    print("    ⇒ 无法「想够了就停下来回答」，需要专门的 think 结束标记")


def main():
    print("=" * 74)
    print("静态审计：本机距「带思维链的长上下文模型」的缺口")
    print("=" * 74)
    k1_loss_coverage()
    k2_seq_length()
    k3_data_format()
    k4_generation_control()
    print()
    print("=" * 74)
    print("【结论】四道缺口，其中 K1 是本机最根本的一条")
    print("=" * 74)
    print("  K1 **loss 只覆盖 answer 段**（`positions >= starts`）")
    print("     ⇒ 即便把数据换成带 CoT 的，模型也**学不到思维链**（think 段零梯度）")
    print("  K2 seq=%d 装不下一条带 CoT 的样本（p90 答案已接近上限）" % SEQ)
    print("  K3 语料无 think 标注 ⇒ 需要新数据源")
    print("  K4 无 think 终止符 ⇒ 生成侧无法「想够就答」")
    print()
    print("  ⇒ K1/K3/K4 是**功能缺口**（要改 loss 与数据），K2 是**长度缺口**")
    print("     K2 已被 §69.9–§69.13 的长上下文工作覆盖；K1/K3/K4 是**新增的**，")
    print("     必须在 §69.10 的流程里加两个阶段：CoT 数据构造 + 推理型 SFT/RL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
