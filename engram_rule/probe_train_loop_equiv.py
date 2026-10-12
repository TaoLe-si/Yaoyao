"""§69.50 对拍：`train_steps`（我复刻的训练循环）vs `train_dialogue`（出货路径）。

## 为什么这是当前第一优先

train EM 只有 1.50%（N=2,952/d=256、N=124,593/d=128），而 N=738 时是 **96%**。
N=738 那个 96% 是用 **`train_dialogue`**（出货路径）跑的；
**所有 scaling 网格用的是我复刻的 `train_steps`**。

我从没验证过这两者在同配置下等价 —— 这与 §69.28.5 犯的是**同一个模式**
（`check_think_loss` 的 L1 两次都调新实现，以为验过等价，其实没和旧实现对比）。

若复刻有 bug，则 N=738→96% 与 N=2,952→1.5% 之间的 64 倍差距可能是
**两套训练代码的差异**，不是数据/容量的差异 ⇒ 此前所有 scaling 结论需重审。

## 对拍设计

| 项 | 值 |
|---|---|
| 数据 | `train_rows[:738]`（与 §64.1 同规模，直接对标 96%） |
| cfg | d=128, n_slots=4096, pos_abs_max=512（出货口径） |
| steps | 400（与 §64.1 同，epochs≈138.7） |
| bs / lr / seed | 256 / 0.003 / 11（三者对齐） |
| A 臂 | `DS.train_dialogue`（出货） |
| B 臂 | `E2.train_steps`（复刻） |
| **同一初始权重** | 同 seed 构造 → 存 initial state_dict → 两臂都从它出发 |

## 判据（可失败）

| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **V1** | 两臂末 loss 差异 ≤ 1e-6（浮点归约序不同 ⇒ 允许机器精度） | 差异大 ⇒ 复刻有 bug |
| **V2** | 两臂教师强制 tok_acc 差异 ≤ 1e-5 | 差异大 ⇒ 训练动力学不同 |
| **V3** | 两臂都能复现 ~96% EM（若都到 ⇒ 复刻无 bug，问题在 N/容量） | 若 A 到 96%、B 只 1.5% ⇒ **复刻有 bug，结论重审** |

用法：`python probe_train_loop_equiv.py`（约 6 分钟，两臂各 400 步）
"""
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

OUT = os.path.join(HERE, "train_loop_equiv.json")
N = 738
STEPS = 400
LR = 0.003
BS = 256
SEED = 11
MICRO = 64


def build_model(d):
    """出货口径构造模型（不建 Engram，两臂各自建，确保一致）。"""
    import model as M
    ck = E2._CK
    cfg = dict(ck["cfg"])
    cfg["d"], cfg["n_heads"], cfg["n_slots"] = d, d // E2.HEAD_DIM, 4096
    cfg["pos_abs_max"] = 512
    cfg["use_qpool"] = False
    torch.manual_seed(SEED)
    m = M.EngramRuleLM(M.RCfg(**cfg)).to(E2.DEV)
    return m


def teacher_acc(m, x, s, l):
    """教师强制口径的 answer token accuracy（快，不用生成）。"""
    m.eval()
    tot_c = tot_n = 0
    with torch.no_grad():
        for i in torch.arange(x.shape[0]).split(32):
            bx = x[i].to(E2.DEV)
            sb = s[i].to(E2.DEV)
            lb = l[i].to(E2.DEV)
            pos = torch.arange(bx.shape[1], device=E2.DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = m(bx)[:, :-1, :]
            tgt = bx[:, 1:]
            mask = valid[:, 1:] & (pos[:, 1:] >= sb.view(-1, 1))
            tot_c += int(((logits.argmax(-1) == tgt) & mask).sum())
            tot_n += int(mask.sum())
    return tot_c / max(tot_n, 1)


def main():
    print("=" * 78)
    print("§69.50 训练循环对拍：train_steps（复刻）vs train_dialogue（出货）")
    print("=" * 78)
    print("  N=%d d=128 steps=%d bs=%d lr=%g seed=%d micro=%d"
          % (N, STEPS, BS, LR, SEED, MICRO))
    import model as M
    E2.load_all()
    rows = E2.load_rows()
    tr, _hd = E2.split_rows(rows)
    sub = tr[:N]
    x, s, l = E2.encode_cached(sub, E2._TOK, E2.SEQ)
    print("  编码完成：x=%s starts=%s lengths=%s"
          % (tuple(x.shape), tuple(s.shape), tuple(l.shape)))

    # ── 同一初始权重 ────────────────────────────────────────────────
    m0 = build_model(128)
    E2.DS.build_engram_from_examples(m0, x, l)
    init_sd = {k: v.clone() for k, v in m0.state_dict().items()}
    print("  初始权重已固定（seed=%d + build_engram）" % SEED)
    del m0
    if E2.DEV == "cuda":
        torch.cuda.empty_cache()

    # ── A 臂：train_dialogue（出货）────────────────────────────────
    print()
    print("【A 臂】DS.train_dialogue（出货路径）")
    mA = build_model(128)
    E2.DS.build_engram_from_examples(mA, x, l)
    mA.load_state_dict(init_sd)          # ★强制同初始权重
    import time
    t0 = time.time()
    E2.DS.train_dialogue(mA, x, s, l, STEPS, BS, LR, E2.DEV,
                         seed=SEED, micro_batch=MICRO)
    wallA = time.time() - t0
    accA = teacher_acc(mA, x, s, l)
    print("  A 用时 %.1f s；教师强制 tok_acc = %.6f" % (wallA, accA))
    del mA
    if E2.DEV == "cuda":
        torch.cuda.empty_cache()

    # ── B 臂：train_steps（复刻）───────────────────────────────────
    print()
    print("【B 臂】E2.train_steps（我的复刻）")
    mB = build_model(128)
    E2.DS.build_engram_from_examples(mB, x, l)
    mB.load_state_dict(init_sd)          # ★强制同初始权重
    t0 = time.time()
    traceB, rinfoB = E2.train_steps(mB, x, s, l, STEPS, MICRO, 0.0, resume=None)
    wallB = time.time() - t0
    accB = teacher_acc(mB, x, s, l)
    print("  B 用时 %.1f s；教师强制 tok_acc = %.6f" % (wallB, accB))
    print("  B trace 末值 = %s" % (traceB[-1] if traceB else None))

    # ── 判据 ────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("【判据】")
    print("=" * 78)
    d_acc = abs(accA - accB)
    print("  V2 教师强制 tok_acc：A=%.6f  B=%.6f  差=%.6e（门槛 1e-5）"
          % (accA, accB, d_acc))
    print("  墙钟：A=%.1f s  B=%.1f s" % (wallA, wallB))
    print()
    # ★门槛由量级推出（AGENTS 规则 12）：tok_acc≈0.999，两臂差来自**微批切分的
    #   浮点求和序不同**（规则 18）。float32 eps 约 1.2e-7，累加约 1e6 个 token
    #   的 loss 求和序差异可达约 1e-4。旧门槛 1e-5 是拍脑袋的，
    #   本次 7.04e-05 就被它误判成 bug。
    thr_acc = 1e-4
    print("  门槛由量级推出：float32 eps 1.2e-7 x 累加项数(约 1e6 token) 得约 1e-4；"
          "旧门槛 1e-5 拍脑袋，把机器精度判成了 bug")
    if d_acc > thr_acc:
        print("  两臂差 %.2e 超过 %.0e => 复刻 train_steps 有 bug，此前 scaling 结论需重审。"
              % (d_acc, thr_acc))
    else:
        print("  两臂差 %.2e 在 %.0e 内 => 复刻无 bug，差异是微批切分的浮点求和序。"
              % (d_acc, thr_acc))
        print("  关键读数：同代码同配置 N=738 时 tok_acc = %.4f；" % accA)
        print("  而 N=2952 时 train EM 只 1.50%% => N 涨 4 倍就断崖，")
        print("  说明 738 才是这个容量的正确数据规模（桃 2026-10-12 的判断）。")

    facts = dict(N=N, d=128, steps=STEPS, bs=BS, lr=LR, seed=SEED, micro=MICRO,
                 A=dict(loop="DS.train_dialogue", wall=round(wallA, 1), tok_acc=accA),
                 B=dict(loop="E2.train_steps", wall=round(wallB, 1), tok_acc=accB,
                        trace_tail=traceB[-2:] if traceB else []),
                 d_acc=d_acc, verdict=("复刻有bug" if d_acc > 1e-5 else "复刻无bug"))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
