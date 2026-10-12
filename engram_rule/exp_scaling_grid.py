"""★实验 D: 二维网格(数据规模 N × 模型容量 D) 拟合幂律 —— scaling law 的**真**证明。

**为什么必须这么做**(§59.5 已写死): 固定 820 条语料、只加容量观察到的"饱和",
**正是 Chinchilla 结论的复现, 不是 scaling law 成立的证据**。要谈 scaling law 至少需要
多个数据规模 × 多个容量点, 且**每点有误差带**。

**预登记(跑之前写死, 见 DESIGN §63)**:

  拟合式  L(N, D) = E + A / N^α + B / D^β
  · L = 在**固定探针集**上的答案 NLL(与 N、D 都无关的同一把尺子)
  · E = 不可约损失(拟合的自由参数, 不固定为 0)

  判据 S1  **α > 0 显著**: 数据越多损失越低, 且 log-log 上近线性
  判据 S2  **β > 0 显著**: 容量越大损失越低
  判据 S3  **E > 0**: 存在不可约项(纯背下 820 条的话 E 应接近 0)
  判据 S4  **拟合质量 R² 够高** 且残差无系统趋势
  否证条件: α 或 β 的置信区间跨 0 ⇒ **不得声称 scaling law**

**口径纪律**:
  · 探针集 = 最后 82 行, **任何一格都不用它训练** ⇒ 各格的 L 可比
  · 训练用前 N 行(N 从 738 里取), 打乱但**按行**切(不切 token)
  · 训练配方与 §59/§62 **逐字相同**(200 步 / bs=256 / lr=0.003 / head_dim=16 /
    micro_batch 按 §48 实测档位 / trim_padding 由 train_dialogue 自己判)
  · 每格落盘一条(generator), gate 覆盖 exp_scaling_grid.py + model.py + dialogue_small.py
  · ★单 seed(桃的指示), 误差带由**网格内的拟合残差**给, 不跑多种子
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
OUT = os.path.join(HERE, "scaling_grid_v1.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
STEPS = 200
LR = 0.003
BS = 256
SEED = 11
HEAD_DIM = 16
N_PROBE = 82                 # 固定探针集(最后 82 行), 任何一格都不训练它
N_POOL = 738                 # 可用训练池 = 820 - 82
N_LIST = (50, 100, 200, 400, 738)
D_LIST = (32, 64, 128)
# 单步墙钟基线(§59/§62 实测): d=64 约 0.48 s/步 ⇒ 200 步约 96 s
REF_SEC_PER_STEP = 0.5


def _gate():
    h = hashlib.sha256()
    for name in ("exp_scaling_grid.py", "model.py", "dialogue_small.py"):
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
    tr_rows = L[2]
    tr_pack = L[4]
    return ck, tok, tr_rows, tr_pack


def mb_for(d):
    """与 §48 实测一致: d<=128 → 64(不试 64 再降半, 见 find_micro_batch 的注释)。"""
    return 64 if d <= 128 else 32


def train_and_probe(ck, tok, tr_rows, tr_pack, n_train, d, steps=STEPS):
    """训练前 n_train 行, 然后在**固定探针集**上量答案 NLL / token acc / EM。"""
    from model import RCfg, EngramRuleLM
    from dialogue_small import (build_engram_from_examples, answer_loss_sums,
                                encode_examples, answer_metrics, evaluate_generations,
                                normalize_text, char_f1)

    rows = tr_rows[:n_train]
    x, s, l = encode_examples(rows, tok, SEQ)
    probe_rows = tr_rows[N_POOL:]
    px, ps, pl = encode_examples(probe_rows, tok, SEQ)

    cfg = dict(ck["cfg"])
    cfg["d"] = d
    cfg["n_heads"] = d // HEAD_DIM
    torch.manual_seed(SEED)
    model = EngramRuleLM(RCfg(**cfg)).to(DEV)
    build_engram_from_examples(model, x, l)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
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
        for i in idx.split(mb_for(d)):
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
    wall = time.time() - t0
    if wall > REF_SEC_PER_STEP * steps * 3.0:
        raise TimeoutError("训练 %d 步(d=%d) 用 %.0fs > %.0fs 闸门"
                           % (steps, d, wall, REF_SEC_PER_STEP * steps * 3.0))

    # 探针: 教师强制(定死的同一把尺子)
    m = answer_metrics(model, px, ps, pl, 32, DEV)
    # 训练子集上的自由生成(报优化状态, 不用于拟合)
    m_tr, outs_tr = evaluate_generations(model, tok, rows, max_new_tokens=90,
                                         device=DEV, context_limit=SEQ,
                                         batch_min_size=8)
    params = sum(p.numel() for p in model.parameters())
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return dict(probe_nll=m["answer_nll"], probe_ppl=m["answer_ppl"],
                probe_tok_acc=m["answer_token_accuracy"],
                probe_first_acc=m["first_answer_token_accuracy"],
                probe_tokens=m["answer_tokens"],
                train_gen_em=m_tr["exact_match"], train_gen_f1=m_tr["char_f1"],
                params=params, wall_sec=round(wall, 1),
                sec_per_step=round(wall / steps, 3), loss_trace=trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--ds", default=None)
    ap.add_argument("--ns", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print("gate = %s (exp_scaling_grid + model + dialogue_small)" % GATE)

    def done(n, d):
        if not os.path.exists(OUT) or a.force:
            return False
        with open(OUT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    if j.get("gate") == GATE and j.get("n") == n and j.get("d") == d:
                        return True
        return False

    ck, tok, tr_rows, tr_pack = load_all()
    ds = tuple(int(v) for v in a.ds.split(",")) if a.ds else D_LIST
    ns = tuple(int(v) for v in a.ns.split(",")) if a.ns else N_LIST
    print("网格: N=%s × d=%s  (%d 格, 每格 %d 步)"
          % (list(ns), list(ds), len(ns) * len(ds), a.steps), flush=True)
    for n in ns:
        for d in ds:
            if done(n, d):
                print("  跳过 N=%-4d d=%-3d (已完成)" % (n, d), flush=True)
                continue
            print("  >>> N=%-4d d=%-3d ..." % (n, d), flush=True)
            rec = train_and_probe(ck, tok, tr_rows, tr_pack, n, d, a.steps)
            rec = dict(gate=GATE, n=n, d=d, steps=a.steps, lr=LR, bs=BS,
                       seed=SEED, head_dim=HEAD_DIM, micro_batch=mb_for(d),
                       n_probe=N_PROBE, **rec)
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print("      probe NLL=%.4f  tok_acc=%.4f  1st=%.4f | train gen_em=%.4f "
                  "| %d 参数 %.0fs (%.3f s/步)"
                  % (rec["probe_nll"], rec["probe_tok_acc"], rec["probe_first_acc"],
                     rec["train_gen_em"], rec["params"], rec["wall_sec"],
                     rec["sec_per_step"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
