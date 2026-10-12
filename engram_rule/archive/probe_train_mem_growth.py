"""★循环代码审计 · 第 1 步: 训练显存是**泄漏**还是**碎片化**?

**为什么现在才查**: `find_micro_batch` 的探测只跑 1 步, 量到 d=64/mb=64 峰值
**1288 MiB**; 但 200 步实跑把 8188 MiB 的卡**占满**。1 步与 200 步之间唯一变的是
"不同长度的张量反复分配" —— `trim_padding=True` 让每个微批的 L 都不同(820 行
行长中位 60、最大 512), 于是分配器面对成百上千种尺寸。

**判据(两个量必须分开读, 只看其中一个必然误判)**:
  · `memory_allocated` —— 真正存活的张量。泄漏会让它**单调上升**。
  · `memory_reserved`  —— 分配器向驱动要的池子。碎片化会让它上升而 allocated 不动。

**自带硬闸**: reserved 一旦超过 HARD_CAP_MIB 立刻抛异常 —— 本脚本自己**不允许**
成为又一次显存事故(规则 19: 大作业前先看门狗)。
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIB = 1 << 20
HARD_CAP_MIB = 3000          # ★ 硬闸: 超过就中止, 绝不把显存占满


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bucket", type=int, default=0,
                    help="把截断长度向上取整到该步长的倍数(0=不取整)。"
                         "形状种类是碎片化的直接来源。")
    a = ap.parse_args()

    import exp_capacity_scaling as ecs
    from model import EngramRuleLM
    from dialogue_small import (batches, _trim_length, answer_loss_sums, build_engram_from_examples)

    ck, tok, tr_rows, ev_rows, tr_pack, ev_pack = ecs.load_data()
    tr_x, tr_s, tr_l = tr_pack
    cfg = dict(ck["cfg"])
    cfg["d"] = a.d
    cfg["n_heads"] = a.d // ecs.HEAD_DIM
    from model import RCfg
    torch.manual_seed(11)
    model = EngramRuleLM(RCfg(**cfg)).to(DEV)
    build_engram_from_examples(model, tr_x, tr_l)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    free0, total = torch.cuda.mem_get_info()
    print("d=%d mb=%d steps=%d | 启动前空闲 %.0f/%.0f MiB"
          % (a.d, a.mb, a.steps, free0 / MIB, total / MIB))

    import model as M
    rope0 = len(M._ROPE_CACHE)
    arng0 = len(M._ARANGE_CACHE)
    lengths_seen = set()
    gen = torch.Generator().manual_seed(11)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, tr_x.shape[0], (ecs.BS,), generator=gen)
        total_ans = (tr_l[idx] - tr_s[idx]).sum().clamp_min(1).to(DEV)
        opt.zero_grad(set_to_none=True)
        for i in idx.split(a.mb):
            bx_full = tr_x[i].to(DEV)
            sb = tr_s[i].to(DEV)
            lb = tr_l[i].to(DEV)
            # ★分桶必须在**原始 512 宽**的张量上切。`_trim_length` 返回的是
            #   已经收窄的视图, 再对它切片只会被 clamp 回原宽度, 分桶等于没做
            #   (第二个同款 bug: 第一次写成 min(当前长度, 取整), 这次是切错对象;
            #    两次的表征都是"读数与不处理逐位相同")。
            keep = int(lb.max())
            if a.bucket > 0:
                keep = -(-keep // a.bucket) * a.bucket
            bx = bx_full[:, :max(keep, 2)]
            # ★记**截断之后**的长度: 记截断前的恒为 512, 量不到形状多样性。
            lengths_seen.add(int(bx.shape[1]))
            pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            tok_loss, first_loss, _, _ = answer_loss_sums(logits, bx[:, 1:], valid, sb)
            (tok_loss + first_loss).div(total_ans).backward()
        opt.step()
        alloc = torch.cuda.memory_allocated() / MIB
        resv = torch.cuda.memory_reserved() / MIB
        if step < 6 or step % 5 == 0 or step == a.steps - 1:
            print("  step %3d  allocated %7.1f  reserved %7.1f  碎片 %6.1f  "
                  "rope缓存 %4d  arange缓存 %4d  见过L %3d 种"
                  % (step, alloc, resv, resv - alloc,
                     len(M._ROPE_CACHE), len(M._ARANGE_CACHE), len(lengths_seen)))
        if resv > HARD_CAP_MIB:
            raise SystemExit(
                "★硬闸触发: reserved %.0f MiB > %d MiB, 在第 %d 步中止"
                % (resv, HARD_CAP_MIB, step))

    print("跑完 %d 步用时 %.1fs; rope 缓存 %d→%d, arange 缓存 %d→%d, 不同 L 共 %d 种"
          % (a.steps, time.time() - t0, rope0, len(M._ROPE_CACHE),
             arng0, len(M._ARANGE_CACHE), len(lengths_seen)))
    print("  实际出现的长度(bucket=%d): %s"
          % (a.bucket, sorted(lengths_seen)[:24]))
    print("最终 allocated %.1f MiB / reserved %.1f MiB (碎片 %.1f MiB)"
          % (torch.cuda.memory_allocated() / MIB, torch.cuda.memory_reserved() / MIB,
             (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / MIB))
    return 0


if __name__ == "__main__":
    sys.exit(main())
