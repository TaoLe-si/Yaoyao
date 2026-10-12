"""§69.51 N 扫描 + 找最优点：**用 train tok_acc 作判据**（不是 loss，不是 probe NLL）。

## 为什么换判据

§69.50 实测：同代码同配置，N=738 时 tok_acc = **99.93%**，N=2,952 时 train EM = **1.50%**。
**N 涨 4 倍就断崖。** 而 loss 在两者都「收敛」（0.0070 vs 0.0078）
⇒ **loss 被大量 token 摊薄，掩盖了「没学会」**。

⇒ 判据必须是 train tok_acc（逐 token 对不对），不是 loss。

## 扫描设计

| N | steps（epochs 恒 40） | 预计墙钟 |
|---:|---:|---:|
| 738 | 116 | 45 s |
| 1,476 | 231 | 90 s |
| 2,952 | 462 | 180 s |
| 5,904 | 924 | 360 s |
| 11,808 | 1,848 | 720 s |
| 23,616 | 3,696 | 1,440 s |

合计约 47 分钟，每档都在 3600 s 上限内 ⇒ 一次跑完、逐档落盘。

## 判据（可失败，先登记）

| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **P1** | tok_acc 随 N 单调降（容量固定，数据越多越记不住） | 若上升 ⇒ 假设反了 |
| **P2** | 存在 N* 使 held tok_acc 最高 | 若单调 ⇒ 无内点最优点 |
| **P3** | tok_acc 首次跌破 0.5 的 N = 容量临界点 | 找不到断崖 ⇒ 容量判断错 |

★推荐工作点取法：**tok_acc 仍 ≥0.9 的最大 N** —— 在「确实学会了」的前提下尽可能多的数据。

用法：`python exp_n_sweep.py`
"""
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

OUT = os.path.join(HERE, "n_sweep.jsonl")
D = 128
EPOCHS = 40.0
BS = 256
LR = 0.003
SEED = 11
MICRO = 64
NS = [738, 1476, 2952, 5904, 11808, 23616]
SEC_PER_STEP = 0.389            # §69.50 实测（N=738 / d=128 / 400 步 = 92.9 s）


def steps_for(n):
    return -(-int(EPOCHS * n) // BS)          # ceil(40N/256)，epochs 恒定


def teacher_acc(m, x, s, l):
    """教师强制口径的 answer token accuracy（全量，不抽样）。"""
    m.eval()
    c = n_ = 0
    with torch.no_grad():
        for i in torch.arange(x.shape[0]).split(32):
            bx = x[i].to(E2.DEV)
            sb = s[i].to(E2.DEV)
            lb = l[i].to(E2.DEV)
            pos = torch.arange(bx.shape[1], device=E2.DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = m(bx)[:, :-1, :]
            mask = valid[:, 1:] & (pos[:, 1:] >= sb.view(-1, 1))
            c += int(((logits.argmax(-1) == bx[:, 1:].to(E2.DEV)) & mask).sum())
            n_ += int(mask.sum())
    return c / max(n_, 1)


def gen_em(m, tok, rows, k=200, seed=12345):
    """生成式 EM + char-F1（抽 k 条，种子固定）。"""
    rnd = random.Random(seed)
    rs = rnd.sample(rows, min(k, len(rows)))
    em, f1s = 0, []
    for r in rs:
        out = E2.DS.greedy_generate(m, tok, r, max_new_tokens=140,
                                    device=E2.DEV, context_limit=512,
                                    temperature=0.0, utf8_safe=True)
        em += int(out["generated"].strip() == out["reference"].strip())
        f1s.append(E2.DS.char_f1(out["generated"], out["reference"]))
    return em, len(rs), (sum(f1s) / len(f1s) if f1s else 0.0)


def main():
    print("=" * 78)
    print("§69.51 N 扫描（判据 = train tok_acc，不是 loss）")
    print("=" * 78)
    print("  d=%d  epochs=%.1f  bs=%d  lr=%g  seed=%d  micro=%d"
          % (D, EPOCHS, BS, LR, SEED, MICRO))
    print("  N 档：%s" % NS)
    print("  steps：%s" % [steps_for(n) for n in NS])
    est = [round(steps_for(n) * SEC_PER_STEP) for n in NS]
    print("  预计各档墙钟：%s s（合计约 %.0f 分钟）"
          % (est, sum(steps_for(n) for n in NS) * SEC_PER_STEP / 60.0))

    import model as M
    E2.load_all()
    rows = E2.load_rows()
    tr, hd = E2.split_rows(rows)
    print("  全量 train %d / held %d" % (len(tr), len(hd)))

    for n in NS:
        st = steps_for(n)
        t0 = time.time()
        sub = tr[:n]
        x, s, l = E2.encode_cached(sub, E2._TOK, E2.SEQ)
        cfg = dict(E2._CK["cfg"])
        cfg["d"], cfg["n_heads"], cfg["n_slots"] = D, D // E2.HEAD_DIM, 4096
        cfg["pos_abs_max"], cfg["use_qpool"] = 512, False
        torch.manual_seed(SEED)
        m = M.EngramRuleLM(M.RCfg(**cfg)).to(E2.DEV)
        E2.DS.build_engram_from_examples(m, x, l)
        trace, _ri = E2.train_steps(m, x, s, l, st, MICRO, 0.0, resume=None)
        acc = teacher_acc(m, x, s, l)
        em_tr, n_tr, f1_tr = gen_em(m, E2._TOK, sub, k=200)
        # held：教师强制口径（抽 2000 条，全量 31,147 生成太慢）
        hx, hs, hl = E2.encode_cached(hd[:2000], E2._TOK, E2.SEQ)
        hacc = teacher_acc(m, hx, hs, hl)
        wall = time.time() - t0
        rec = dict(n=n, d=D, steps=st, epochs=round(BS * st / n, 3),
                   train_tok_acc=acc, train_em=em_tr, train_em_n=n_tr,
                   train_f1=f1_tr, held_tok_acc=hacc,
                   trace_tail=trace[-2:] if trace else [], wall_sec=round(wall, 1))
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print()
        print("  N=%-6d steps=%-5d  train_tok_acc=%.4f  train_EM=%d/%d  F1=%.4f  "
              "held_tok_acc=%.4f  %.0fs"
              % (n, st, acc, em_tr, n_tr, f1_tr, hacc, wall), flush=True)
        del m
        if E2.DEV == "cuda":
            torch.cuda.empty_cache()

    # ── 汇总 + 最优点 ──────────────────────────────────────────────
    print()
    print("=" * 78)
    print("【汇总 + 最优点】")
    print("=" * 78)
    recs = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    print("  %-8s %-7s %-13s %-10s %-9s %-13s"
          % ("N", "steps", "train_tokacc", "train_EM", "train_F1", "held_tokacc"))
    for r in recs:
        print("  %-8d %-7d %-13.4f %-10s %-9.4f %-13.4f"
              % (r["n"], r["steps"], r["train_tok_acc"],
                 "%d/%d" % (r["train_em"], r["train_em_n"]),
                 r["train_f1"], r["held_tok_acc"]))
    mono = all(recs[i]["train_tok_acc"] >= recs[i + 1]["train_tok_acc"]
               for i in range(len(recs) - 1))
    print()
    print("  P1（tok_acc 随 N 单调降）：%s" % ("成立" if mono else "不成立，需复核"))
    cliff = next((r["n"] for r in recs if r["train_tok_acc"] < 0.5), None)
    print("  P3 断崖点（tok_acc 首次 <0.5）= %s"
          % (cliff or "未出现（%d 档全部 >=0.5）" % len(recs)))
    best_h = max(recs, key=lambda r: r["held_tok_acc"])
    print("  P2 held_tok_acc 最高的 N = %d（%.4f）" % (best_h["n"], best_h["held_tok_acc"]))
    print()
    ok = [r for r in recs if r["train_tok_acc"] >= 0.9]
    print("  tok_acc >= 0.9 的 N 档 = %s" % [r["n"] for r in ok])
    if ok:
        w = max(ok, key=lambda r: r["n"])
        print("  ⇒ 推荐工作点 **N=%d**（steps=%d, tok_acc=%.4f, held_tok_acc=%.4f）"
              % (w["n"], w["steps"], w["train_tok_acc"], w["held_tok_acc"]))
    else:
        print("  ⇒ 没有任何档 tok_acc >= 0.9 ⇒ 6 档都超出容量，需往下扫（N<738）")
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
