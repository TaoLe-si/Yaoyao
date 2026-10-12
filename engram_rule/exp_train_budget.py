"""★实验 E: 训练预算扫描 —— 固定 N=738、d=128, 只加步数。

**它一次回答两个问题**:
  (1) §63.7 自己写下的竞争解释: 「容量项 β 不显著」可能是**大容量没训够**,
      不是容量真的没用。若加步数后 train/probe 都明显改善, §63 的 β 结论要作废重写。
  (2) 目标里「训练集达到优秀水平」—— 当前 d=128 是 EM 0.7866(可判子集 0.8855),
      还剩 175 行错。训练预算是**唯一还没动过的杠杆**(§58.4 列的四条都不推荐)。

**预登记判据(跑之前写死, 见 DESIGN §64)**:
  T1  train 可判子集 EM 从 0.8855 **上升到 ≥ 0.95** ⇒ 训练预算有效
  T2  train char-F1 从 0.9283 **上升到 ≥ 0.97**
  T3  第一步就错(lcp=0)从 0.0073 降到 **0**
  T4  ★对 §63 的复核: 探针 NLL 是否继续下降(若降 ⇒ §63 的"加容量变差"是**步数不够**的假象)
  否证: 四个判据都不动 ⇒ 预算不是瓶颈, 剩下的错是**数据/架构**问题, 加步数是浪费

**口径纪律**:
  · N=738(全训练池)、d=128、seed=11、bs=256、lr=0.003、head_dim=16、micro_batch=64
    —— 与 §59/§62/§63 **逐字相同**, 唯一变的是 steps
  · `max_new=400` 报**可判子集** EM(§60.5 口径修正), `max_new=90` 报全集快口径
  · 逐格落盘(generator); gate 覆盖 exp_train_budget.py + model.py + dialogue_small.py
  · ★单 seed(桃的指示); 每格报墙钟, 闸门按步数缩放
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "train_budget_v1.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
LR = 0.003
BS = 256
SEED = 11
HEAD_DIM = 16
N_TRAIN = 738               # 全训练池(= 820 - 82 探针)
N_PROBE = 82                # 固定探针集, 与 §63 同一批行
D = 128
MB = 64
MAX_NEW_FULL = 400          # §60.5 口径
MAX_NEW_FAST = 90
N_FULL_ROWS = 200           # 全量 EM 的行数上限(取满可判子集)
STEP_LIST = (200, 400, 800, 1600)
# 基线: d=128 实测 0.48 s/步(§59) ⇒ 闸门取 3 倍
REF_SEC_PER_STEP = 0.5
# §59 的锚点(同一 d=128 / seed=11 / 200 步)
ANCHOR_EM_JUDGEABLE = 0.8855
ANCHOR_F1 = 0.9283
ANCHOR_LCP0 = 0.0073


def _gate():
    h = hashlib.sha256()
    for name in ("exp_train_budget.py", "model.py", "dialogue_small.py"):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:12]


GATE = _gate()


def load_all():
    import exp_capacity_scaling as ecs
    from tokenizers import Tokenizer
    ck = torch.load(ecs.V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    L = ecs.load_data()
    return ck, tok, L[2], L[4]


@torch.no_grad()
def probe_nll(model, px, ps, pl):
    """固定探针集上的答案 NLL / token acc —— 与 §63 同一把尺子。"""
    from dialogue_small import answer_metrics
    return answer_metrics(model, px, ps, pl, 32, DEV)


@torch.no_grad()
def gen_full(model, tok, rows, max_new):
    """**出货的** evaluate_generations(批生成), 逐行拿 generated/em/f1。"""
    import dialogue_small as ds
    m, outs = ds.evaluate_generations(
        model, tok, rows, max_new_tokens=max_new, device=DEV,
        context_limit=SEQ, temperature=0.0, batch_min_size=8)
    per = []
    for o in outs:
        em = float(ds.normalize_text(o["generated"])
                   == ds.normalize_text(o["reference"]))
        per.append(dict(generated=o["generated"], reference=o["reference"],
                        gen_em=em, gen_f1=ds.char_f1(o["generated"], o["reference"])))
    return m, per


def run(ck, tok, tr_rows, tr_pack, steps):
    from model import RCfg, EngramRuleLM
    from dialogue_small import (build_engram_from_examples, answer_loss_sums,
                                encode_examples)

    rows = tr_rows[:N_TRAIN]
    x, s, l = encode_examples(rows, tok, SEQ)
    probe_rows = tr_rows[N_TRAIN:]
    px, ps, pl = encode_examples(probe_rows, tok, SEQ)

    cfg = dict(ck["cfg"])
    cfg["d"] = D
    cfg["n_heads"] = D // HEAD_DIM
    torch.manual_seed(SEED)
    model = EngramRuleLM(RCfg(**cfg)).to(DEV)
    build_engram_from_examples(model, x, l)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    # ★total_steps 必须等于 steps —— 改步数就改 LR 计划, 这正是"预算"的一部分
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                               pct_start=0.1)
    g = torch.Generator().manual_seed(SEED)
    model.train()
    t0 = time.time()
    trace = []
    for step in range(steps):
        idx = torch.randint(0, x.shape[0], (BS,), generator=g)
        norm = (l[idx] - s[idx]).sum().clamp_min(1).to(DEV)
        opt.zero_grad(set_to_none=True)
        sl = torch.zeros((), dtype=torch.float32, device=DEV)
        for i in idx.split(MB):
            bx_full = x[i].to(DEV)
            sb = s[i].to(DEV)
            lb = l[i].to(DEV)
            bx = bx_full[:, :max(int(lb.max()), 2)]
            pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            tl, fl, _, _ = answer_loss_sums(logits, bx[:, 1:], valid, sb)
            ((tl + fl) / norm).backward()
            sl += tl.detach()
        opt.step()
        sched.step()
        if step == 0 or (step + 1) % max(1, steps // 4) == 0:
            trace.append(float(sl / norm))
            print("      step %d/%d loss %.4f (%.0fs)"
                  % (step + 1, steps, trace[-1], time.time() - t0), flush=True)
    wall = time.time() - t0
    cap = REF_SEC_PER_STEP * steps * 3.0
    if wall > cap:
        raise TimeoutError("训练 %d 步用 %.0fs > %.0fs 闸门" % (steps, wall, cap))

    pm = probe_nll(model, px, ps, pl)
    # 全量口径(max_new=400) —— 报"§60.5 修正后的成绩"
    sub = rows[:N_FULL_ROWS]
    m400, per400 = gen_full(model, tok, sub, MAX_NEW_FULL)
    # 快口径(max_new=90)全集 —— 与 §59 表格可比
    m90, per90 = gen_full(model, tok, rows, MAX_NEW_FAST)
    params = sum(p.numel() for p in model.parameters())
    ckpt = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_d128_s%d_st%d.pt"
                         % (SEED, steps))
    torch.save(dict(cfg=cfg, state_dict=model.state_dict(),
                    tokenizer=ck["tokenizer"],
                    extra=dict(steps=steps, lr=LR, seed=SEED, gate=GATE,
                               wall_sec=round(wall, 1), d=D)), ckpt)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return dict(gate=GATE, d=D, steps=steps, lr=LR, seed=SEED, n_train=N_TRAIN,
                n_probe=N_PROBE, params=params,
                wall_sec=round(wall, 1), sec_per_step=round(wall / steps, 3),
                loss_trace=trace,
                probe_nll=pm["answer_nll"], probe_tok_acc=pm["answer_token_accuracy"],
                probe_first_acc=pm["first_answer_token_accuracy"],
                full_em_400=m400["exact_match"], full_f1_400=m400["char_f1"],
                full_n=len(per400),
                full_rows=[dict(generated=r["generated"], reference=r["reference"],
                                gen_em=r["gen_em"], gen_f1=r["gen_f1"])
                           for r in per400],
                fast_em_90=m90["exact_match"], fast_f1_90=m90["char_f1"],
                ckpt=ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=list(STEP_LIST))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print("gate = %s (exp_train_budget + model + dialogue_small)" % GATE)

    def done(st):
        if not os.path.exists(OUT) or a.force:
            return False
        with open(OUT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    if j.get("gate") == GATE and j.get("steps") == st:
                        return True
        return False

    ck, tok, tr_rows, tr_pack = load_all()
    print("扫描 steps=%s (N=%d, d=%d, seed=%d)"
          % (a.steps, N_TRAIN, D, SEED), flush=True)
    for st in a.steps:
        if done(st):
            print("  跳过 steps=%d (已完成)" % st, flush=True)
            continue
        print("  >>> steps=%d ..." % st, flush=True)
        rec = run(ck, tok, tr_rows, tr_pack, st)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("      probe NLL=%.4f | 全量(400,%d行) EM=%.4f f1=%.4f | "
              "快口径(90,全集) EM=%.4f f1=%.4f | %.0fs (%.3f s/步)"
              % (rec["probe_nll"], rec["full_n"], rec["full_em_400"],
                 rec["full_f1_400"], rec["fast_em_90"], rec["fast_f1_90"],
                 rec["wall_sec"], rec["sec_per_step"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
