# -*- coding: utf-8 -*-
"""§69.36 从满步成品**接着训 50 步**，只看 loss 是否继续下降。

── 桃的要求（2026-10-12）─────────────────────────────────────────────
「基于我本人的判断，仍然是模型欠拟合，先停掉你当前的训练，从刚才完整的存档权重
  继续训 50 步，观察 loss 是否继续下降」

── 为什么这个判据能区分「欠拟合」与「已收敛」────────────────────────
满步权重是在 19,468 步、OneCycle 计划**跑满**的状态下存的 ⇒ 此刻 lr 已降到
OneCycle 的末端（约 0）。若在此基础上再训 50 步 loss 仍明显下降 ⇒ 训练**未收敛**，
即桃说的欠拟合成立；若 loss 基本不动 ⇒ 已收敛，那自由生成的差就不是训练量问题。

★★**必须避免的自欺**：OneCycle 的 `pct_start=0.1` 意味着新起一个 50 步的计划时，
  前 10% 是**升温期**（lr 从 max/div 爬到 max_lr）。若我用默认参数重训，
  loss 下降可能只是 lr 爬升带来的，**不能证明原轨迹未收敛**。
  ⇒ 本脚本用 **恒定小 lr**（不建 OneCycle），这样 loss 的变化只反映
    「在满步权重这点上还有多少可降的余量」，不被 lr 曲线污染。

── 判据（先登记，可失败）───────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **V1** | 50 步内 loss **继续下降**（末 10 步均值 < 首 10 步均值） | 若上升或持平 ⇒ 已收敛，欠拟合判断被否 |
| **V2** | 下降幅度要报出来，且标注 lr 口径 | 不给幅度 ⇒ 无法判断是不是 lr 爬升的假象 |
| **V3** | 用**同一批数据同一 shuffle 序列** ⇒ 两段可比 | 换数据 ⇒ 下降可能来自抽样波动 |

★单 seed、50 步 ⇒ **无种子极差**，不报置信区间（AGENTS 规则 16）。

用法：`python exp_extra50.py [--lr 3e-5] [--steps 50]`
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

CKPT = os.path.join(HERE, "engram_rule_scaling_v2_n124593_d128_wd0p0.pt")
# ★文件名必须含步数：否则 500 步那次会把 50 步的读数**覆盖掉**，
#   而 50 步是「余量 −1.48%（超出噪声）」的唯一原始证据。
#   （2026-10-12 就吃过这个：extra50.json 里现在只剩 500 步那份，50 步只留在旧日志里。）
OUT = os.path.join(HERE, "extra_continue.json")
BASE_LR = 0.003                     # 满步时用的 max_lr，作对照
CONST_LR = 3e-5                     # ★恒定小 lr：0.003 的 1%，避免 OneCycle 升温假象


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=CONST_LR)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--wd", type=float, default=0.0, help="与满步成品同口径 ⇒ 0.0")
    a = ap.parse_args()

    print("=" * 78)
    print("§69.36 从满步成品接着训 %d 步，看 loss 是否继续下降" % a.steps)
    print("=" * 78)
    print("  满步成品 = %s" % os.path.basename(CKPT))
    print("  该格满步时 lr = %.4f（OneCycle max_lr，跑满后末端 lr≈0）" % BASE_LR)
    print("  本次恒定 lr = %.2e（max_lr 的 %.2f%%）—— ★不建 OneCycle，避免升温假象"
          % (a.lr, 100 * a.lr / BASE_LR))
    print("  weight_decay = %.4f（与满步同口径）" % a.wd)

    E2.load_all()
    rows = E2.load_rows()
    train_rows, _held = E2.split_rows(rows)
    print("  train %d 行" % len(train_rows))

    # ── 构造模型 + 载入满步权重 + 建 Engram 表 ──────────────────────
    import model as M                  # ★规则 7：E2 没导出 M
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])              # ★逐字用成品自带 cfg，一个字段都不重建
    torch.manual_seed(E2.SEED)
    m = M.EngramRuleLM(M.RCfg(**cfg)).to(E2.DEV)
    m.load_state_dict(ck["state_dict"])   # ★strict=True，形状不符必须炸
    x, s, l = E2.encode_cached(train_rows, E2._TOK, E2.SEQ)
    info = E2.DS.build_engram_from_examples(m, x, l)
    print("  建 Engram：alive %d / %d (%.4f)"
          % (info["alive_slots"], info["n_slots"], info["alive_ratio"]))
    assert info["alive_slots"] > 0, "alive 全 False ⇒ 注意力被乘 0，测的是残壳"

    # ── 训 50 步，逐步记 loss ───────────────────────────────────────
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    # ★不用 OneCycle：恒定 lr，loss 的变化才只反映「满步点上还有多少余量」
    gen = torch.Generator().manual_seed(E2.SEED)     # ★V3：固定 shuffle 序列
    m.train()
    # ★设备对齐：x/s/l 在 CPU，模型在 cuda。`pos` 是 cuda、`l[i]` 是 cpu ⇒ 比较报错。
    #   逐微批把三个张量搬上设备（micro_batch=64 ⇒ 每次只搬一小片，量可忽略）。
    trace = []
    t0 = __import__("time").time()
    for step in range(a.steps):
        idx = torch.randint(0, x.shape[0], (E2.BS,), generator=gen)
        norm = (l[idx] - s[idx]).sum().clamp_min(1).to(E2.DEV)
        opt.zero_grad(set_to_none=True)
        tot = torch.zeros((), dtype=torch.float32, device=E2.DEV)
        for i in idx.split(E2.MICRO_BATCH):
            bx = x[i].to(E2.DEV)
            sb = s[i].to(E2.DEV)
            lb = l[i].to(E2.DEV)
            bx = E2.DS._trim_length(bx, lb) if cfg.get("bucket_step", 0) > 0 else bx
            pos = torch.arange(bx.shape[1], device=E2.DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = m(bx)[:, :-1, :]
            tl, _fl, _na, _nf = E2.DS.answer_loss_sums(logits, bx[:, 1:], valid, sb)
            (tl / norm).backward()
            tot += tl.detach()
        opt.step()
        val = float(tot / norm)
        trace.append(val)
        if (step + 1) % 10 == 0 or step == 0:
            print("  step %3d/%d  loss %.6f   (%.1f s)"
                  % (step + 1, a.steps, val, __import__("time").time() - t0), flush=True)

    # ── V1/V2 判据 ──────────────────────────────────────────────────
    n = len(trace)
    k = max(5, n // 5)
    head = sum(trace[:k]) / k
    tail = sum(trace[-k:]) / k
    delta = tail - head
    print()
    print("=" * 78)
    print("【V1】loss 是否继续下降")
    print("=" * 78)
    print("  首 %d 步均值 = %.6f" % (k, head))
    print("  末 %d 步均值 = %.6f" % (k, tail))
    print("  变化 = %+.6f  (%+.3f%%)" % (delta, 100 * delta / head))
    print("  单步序列：%s" % " ".join("%.4f" % v for v in trace[:10]), "...")
    # ★V1 的**幅度**门槛（2026-10-12 修，此前只比符号 ⇒ 误报）：
    #   旧判据 `if delta < 0` 在 500 步那次把 −0.023%（纯噪声）判成「loss 继续下降」。
    #   单步抖动来自 bs=256 随机抽样（实测 1.90~2.13），所以门槛必须由**抖动幅度**推出：
    #   取全程单步标准差 σ，判「下降」要求 |Δ| ≥ 2σ/√(n/k)（两段均值差的标准误量级）。
    import statistics as _st
    sd = _st.pstdev(trace)
    se = sd / (k ** 0.5) if k > 0 else float("inf")     # 每段均值的标准误
    thr = 2 * se
    print("  单步标准差 σ = %.6f  每段均值标准误 = %.6f  判据门槛 2σ/√k = %.6f"
          % (sd, se, thr))
    print("  |Δ| = %.6f  vs 门槛 %.6f  ⇒  %s"
          % (abs(delta), thr, "超出噪声" if abs(delta) >= thr else "**在噪声内**"))
    if delta < 0 and abs(delta) >= thr:
        print("  ⇒ **loss 显著继续下降** ⇒ 满步点上仍有可降余量 ⇒ 支持「欠拟合」判断")
    elif delta < 0:
        print("  ⇒ loss 降了但**幅度在噪声内**（%.6f < %.6f）⇒ 判**已收敛**，"
              "「欠拟合」判断在此口径下不成立" % (abs(delta), thr))
    else:
        print("  ⇒ loss 未下降（%+.6f）⇒ **已收敛**，「欠拟合」判断在此口径下被否" % delta)
    print("  ★口径：恒定 lr=%.2e，**不是** OneCycle ⇒ 下降不是升温带来的假象" % a.lr)
    print("  ★单 seed、%d 步 ⇒ 无种子极差，不报置信区间（规则 16）" % a.steps)

    # ── 落盘续训后的权重（供 run_qa_tables 用它跑同一套抽样 200）───
    out_ck = os.path.join(HERE, "engram_rule_extra%d_lr%g.pt"
                          % (a.steps, a.lr))
    torch.save(dict(cfg=cfg, state_dict=m.state_dict(),
                    tokenizer=ck.get("tokenizer"),
                    extra=dict(base_ckpt=os.path.basename(CKPT),
                               base_steps=int(ck["cell"]["steps"]),
                               extra_steps=a.steps, const_lr=a.lr, wd=a.wd,
                               head=head, tail=tail, delta=delta)), out_ck)
    print("  已写出续训权重 %s" % os.path.basename(out_ck))

    facts = dict(ckpt=os.path.basename(CKPT), base_lr=BASE_LR, const_lr=a.lr,
                 wd=a.wd, steps=a.steps, bs=E2.BS, micro=E2.MICRO_BATCH, seed=E2.SEED,
                 trace=trace, head=head, tail=tail, delta=delta,
                 sd=sd, se=se, threshold=thr,
                 verdict=("converged" if not (delta < 0 and abs(delta) >= thr)
                          else "still_improving"),
                 alive=info["alive_slots"], wall=round(__import__("time").time() - t0, 1),
                 out_ckpt=os.path.basename(out_ck))
    # ★读数文件按步数与 lr 分开，保住 50 步那份（曾被 500 步覆盖）
    out_json = os.path.join(HERE, "extra_%d_lr%g.json" % (a.steps, a.lr))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print("  已写出 %s" % os.path.basename(out_json))
    print("  （另存一份到 extra_continue.json 供旧引用）")
    import shutil
    shutil.copyfile(out_json, os.path.join(HERE, "extra_continue.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
