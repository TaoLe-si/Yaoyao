"""★实验 A: 固定数据/步数/seed, 只加模型容量 d, 验"容量受限导致遗忘"。

**这是跑实验的脚本, 只产出原始读数**; 判决全部在 `probe_capacity_scaling.py` 里,
按 DESIGN §48 的预登记规则执行。两者分开是为了不让"跑"的人顺手改判据。

────────────────────────────────────────────────────────────────────────
**口径纪律 (规则 23: 锚点必须与目标口径一致)**

  · 数据从 v11 checkpoint 的 `raw_dialogue_rows` 重建, 并对 eval 侧**逐元素**断言。
  · LOO 分层的**行子集**必须与 §46 (`probe_opener_learnability.py`) **完全一致**,
    否则 R1/R2 不能和 §46 的 0.1832 / 0.7653 直接比。
  · Δ 用 §42 (`probe_delta_conditioning.py`) 的口径: K=1 主口径 + K=all 次口径,
    R=20 个置换, 生成器种子 29, 只在 `fit_subset` 挑出的可比行上算。

**只追加, 不覆盖**: 每次 run 一行 JSON 写进 `capacity_scaling_v1.jsonl`。
默认断点续跑(已存在的 (arm,d,seed,lr) 直接跳过), 所以中断不会毁掉已付的算力。
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
OUT = os.path.join(HERE, "capacity_scaling_v1.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
BS = 256
EVAL_BS = 32
R_PERM = 20
PERM_SEED = 29
HEAD_DIM = 16                     # ★ §48: head_dim 恒为 16, n_heads 随 d 变
D_MAIN = (64, 128, 256)
SEEDS = (11, 23, 37)
PRECHECK_SEEDS = (11, 23)          # ★2 个 seed 才给得出种子极差; 1 个 seed 依规则 12 无法判 lr 敏感性
D_PRECHECK = 128
LR_PRECHECK = (0.001, 0.003, 0.01)


# --------------------------------------------------------------------- 数据

def load_data():
    from tokenizers import Tokenizer
    from dialogue_small import (split_records, encode_examples, format_prefix,
                                format_record)
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr_rows, ev_rows, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                        seed=29, return_meta=True)
    tr_x, tr_s, tr_l = encode_examples(tr_rows, tok, SEQ)
    ev_x, ev_s, ev_l = encode_examples(ev_rows, tok, SEQ)
    assert bool((ev_x == ck["eval_tokens"]).all()), "S1 eval tokens 重建不一致"
    assert ev_s.tolist() == ck["extra"]["eval_answer_starts"], "S2 eval starts 不一致"
    assert ev_l.tolist() == ck["extra"]["eval_lengths"], "S3 eval lengths 不一致"
    return ck, tok, tr_rows, ev_rows, (tr_x, tr_s, tr_l), (ev_x, ev_s, ev_l)


def loo_and_subset(tok, rows, max_len):
    """与 §46 完全一致的行子集 + LOO 频次。口径变了 R1/R2 就不能和 §46 比。"""
    # ⚠ format_prefix/format_record 必须在这里 import: load_data 里 import 的只是
    #   那个函数的局部名, 模块级拿不到(本轮已踩过一次 NameError)。
    from dialogue_small import format_prefix, format_record

    def plen(r):
        return len(tok.encode(format_prefix(r), add_special_tokens=False).ids)

    def rlen(r):
        return len(tok.encode(format_record(r), add_special_tokens=False).ids)

    def opener(r):
        p = plen(r)
        f = rlen(r)
        return f if f <= p else tok.encode(format_record(r), add_special_tokens=False).ids[p]

    top = [opener(r) for r in rows]
    cnt = Counter(top)
    loo = torch.tensor([float(cnt[t] - 1) for t in top])       # ★ 留一
    pl = [plen(r) for r in rows]
    al = [max(0, rlen(rows[i]) - pl[i] - 1) for i in range(len(rows))]
    mx = max(al)
    keep = [i for i in range(len(rows)) if pl[i] + mx + 1 <= max_len]
    return loo[keep], keep, al, mx


@torch.no_grad()
def score_rows(model, rows, tok, max_len):
    from dialogue_small import encode_examples
    x, starts, _ = encode_examples(rows, tok, max_len)
    f1, hit = [], []
    for i in range(0, x.shape[0], EVAL_BS):
        xb = x[i:i + EVAL_BS].to(DEV)
        sb = starts[i:i + EVAL_BS].to(DEV)
        lg = model(xb).float()
        bi = torch.arange(xb.shape[0], device=DEV)
        lp = F.log_softmax(lg, dim=-1)
        f1.append(lp[bi, sb - 1].gather(1, xb[bi, sb].unsqueeze(1)).squeeze(1).cpu())
        hit.append((lg[bi, sb - 1].argmax(-1) == xb[bi, sb]).float().cpu())
    return torch.cat(f1), torch.cat(hit)


def delta_k1(model, rows, tok, max_len):
    """§42 口径: mean[S_1(q,a) - S_1(q, a_π)], R=20 个置换, 只在可比行上。"""
    n = len(rows)
    own, _ = score_rows(model, rows, tok, max_len)
    acc = torch.zeros(n)
    g = torch.Generator().manual_seed(PERM_SEED)
    for _ in range(R_PERM):
        perm = torch.randperm(n, generator=g).tolist()
        pf, _ = score_rows(model, [{"question": rows[i]["question"],
                                    "answer": rows[perm[i]]["answer"],
                                    "history": []} for i in range(n)], tok, max_len)
        acc += pf
    d = own - acc / R_PERM
    return float(d.mean()), float(d.std()) / (n ** 0.5), n


# --------------------------------------------------------------------- 单次 run

def build_cfg(base, d):
    from model import RCfg
    cfg = dict(base)
    cfg["d"] = d
    cfg["n_heads"] = d // HEAD_DIM
    return RCfg(**cfg)


MIN_MICRO = 4
PROBE_STEP_TIMEOUT_S = 8.0


def start_micro_batch(d):
    """按已测的显存-宽度关系定起点, **不盲试**。

    实测(§48 可行性探针): micro_batch=64 时峰值 d=64 → 2448 MiB, d=128 → 4709 MiB,
    近似线性 `peak ≈ 189 + 35.3·d`。外推 d=256 → **9226 MiB > 8188 MiB 物理上限**。
    ⇒ d=256 必须从 micro_batch=32 起。

    ★为什么不"试 64 再降半": d=256 在 mb=64 下**不会抛 OOM**, 而是显存打满后
      Windows WDDM 开始换页 —— 进程卡住、GPU 100%、不返回。实测卡了数分钟没跑完
      (这是 §40 那次事故的同一堵墙)。盲试 = 把事故重演一遍。
    """
    return 64 if d <= 128 else 32


def make_model(ck, d, seed, tr_x, tr_lengths):
    from model import EngramRuleLM
    from dialogue_small import build_engram_from_examples
    torch.manual_seed(seed)
    m = EngramRuleLM(build_cfg(ck["cfg"], d)).to(DEV)
    build_engram_from_examples(m, tr_x, tr_lengths)
    return m


def find_micro_batch(ck, d, seed, tr_pack):
    """花 1 步试出这一档跑得动的 micro_batch, 再正式训 200 步。

    两条失败判据(缺一不可):
      · 抛 OutOfMemoryError  → 降半重来
      · **1 步超过 PROBE_STEP_TIMEOUT_S 秒** → 视同失败。显存打满后的换页抖动
        不抛异常, 只表现为"极慢", 墙钟闸才能抓到它。
    ★切分微批梯度精确等价(见 train_dialogue 的 docstring + C6 实测), 等效 bs 仍 256。
    """
    tr_x, tr_s, tr_l = tr_pack
    from dialogue_small import train_dialogue
    mb = start_micro_batch(d)
    last = float("nan")
    while mb >= MIN_MICRO:
        m = make_model(ck, d, seed, tr_x, tr_l)
        try:
            if DEV == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            train_dialogue(m, tr_x, tr_s, tr_l, 1, BS, 0.003, DEV, seed=seed,
                           first_token_weight=1.0, micro_batch=mb)
            el = time.time() - t0
            last = el
            peak = torch.cuda.max_memory_allocated() / (1 << 20) if DEV == "cuda" else 0.0
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()
            if el > PROBE_STEP_TIMEOUT_S:
                raise TimeoutError("1 步 %.1fs > %.1fs, 疑似换页抖动"
                                   % (el, PROBE_STEP_TIMEOUT_S))
            return mb, peak, el
        except (torch.cuda.OutOfMemoryError, TimeoutError) as exc:
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()
            if mb == MIN_MICRO:
                raise
            nxt = mb // 2
            print("      d=%d micro_batch=%d 失败(%s) -> 降到 %d"
                  % (d, mb, type(exc).__name__, nxt), flush=True)
            mb = nxt
    raise RuntimeError("micro_batch 降到下限仍跑不动: d=%d (最后 1 步 %.1fs)" % (d, last))


def run_one(arm, d, seed, lr, ck, tok, tr_rows, ev_rows, tr_pack, ev_pack,
            tr_loo, tr_keep, ev_loo, ev_keep):
    from dialogue_small import train_dialogue, answer_metrics
    tr_x, tr_s, tr_l = tr_pack
    ev_x, ev_s, ev_l = ev_pack

    mb, probe_peak, probe_sec = find_micro_batch(ck, d, seed, tr_pack)
    model = make_model(ck, d, seed, tr_x, tr_l)          # 正式 run 用干净模型
    tot = sum(p.numel() for p in model.parameters())
    en = sum(p.numel() for n_, p in model.named_parameters() if n_.startswith("engram"))

    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_dialogue(model, tr_x, tr_s, tr_l, 200, BS, lr, DEV, seed=seed,
                   first_token_weight=1.0, micro_batch=mb)
    wall = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / (1 << 20)) if DEV == "cuda" else 0.0

    m_tr = answer_metrics(model, tr_x, tr_s, tr_l, EVAL_BS, DEV)
    m_ev = answer_metrics(model, ev_x, ev_s, ev_l, EVAL_BS, DEV)

    # R1 / R2: 与 §46 同一行子集(keep)、同一分层口径
    sub = [{"question": tr_rows[i]["question"], "answer": tr_rows[i]["answer"],
            "history": []} for i in tr_keep]
    _, hit = score_rows(model, sub, tok, SEQ)
    m0 = tr_loo == 0
    m4 = tr_loo >= 4
    r1, n1 = float(hit[m0].mean()), int(m0.sum())
    r2, n2 = float(hit[m4].mean()), int(m4.sum())
    se1 = (r1 * (1 - r1) / n1) ** 0.5 if n1 else float("nan")
    se2 = (r2 * (1 - r2) / n2) ** 0.5 if n2 else float("nan")

    # R4: Δ (§42 口径)。train 用可比子集, eval 用可比子集。
    dtr_rows = [{"question": tr_rows[i]["question"], "answer": tr_rows[i]["answer"],
                 "history": []} for i in tr_keep]
    dev_rows = [{"question": ev_rows[i]["question"], "answer": ev_rows[i]["answer"],
                 "history": []} for i in ev_keep]
    d_tr, se_tr, n_tr = delta_k1(model, dtr_rows, tok, SEQ)
    d_ev, se_ev, n_ev = delta_k1(model, dev_rows, tok, SEQ)

    rec = dict(arm=arm, d=d, seed=seed, lr=lr, steps=200, bs=BS, seq=SEQ,
               n_heads=d // HEAD_DIM, head_dim=HEAD_DIM, micro_batch=mb,
               params_total=tot, params_engram=en, engram_pct=100.0 * en / tot,
               wall_sec=round(wall, 1), peak_mem_mib=round(peak, 1),
               probe_peak_mib=round(probe_peak, 1), probe_step_sec=round(probe_sec, 2),
               train=m_tr, eval=m_ev,
               r1_loo0_acc=r1, r1_n=n1, r1_se=se1,
               r2_loo4_acc=r2, r2_n=n2, r2_se=se2,
               delta_train=d_tr, delta_train_se=se_tr, delta_train_n=n_tr,
               delta_eval=d_ev, delta_eval_se=se_ev, delta_eval_n=n_ev,
               n_rows=int(hit.numel()))
    # ★逐行命中率 + LOO 频次: 主判据 S(连续回归斜率) 要用**全部 819 个点**,
    #   只存分层均值就只能算原来功效不足的 R3。写在这里, 分析脚本自己算任意分层。
    rec["hit_all"] = [round(float(v), 4) for v in hit.tolist()]
    rec["loo_all"] = [int(v) for v in tr_loo.tolist()]
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return rec


# --------------------------------------------------------------------- 驱动

def done_keys():
    if not os.path.exists(OUT):
        return set()
    ks = set()
    with open(OUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ks.add((r["arm"], r["d"], r["seed"], r["lr"]))
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("precheck", "main"), required=True)
    ap.add_argument("--lrs", default=None,
                    help="逗号分隔, 只跑这些 lr (例如 0.003); 默认跑该 arm 的全部计划")
    ap.add_argument("--reset", action="store_true", help="清空 JSONL(仅在确认无有效数据时用)")
    a = ap.parse_args()

    if a.reset:
        if os.path.exists(OUT):
            raise SystemExit("--reset 会删除 %s, 请手动确认后先自行删除" % OUT)

    ck, tok, tr_rows, ev_rows, tr_pack, ev_pack = load_data()
    tr_loo, tr_keep, _, _ = loo_and_subset(tok, tr_rows, SEQ)
    ev_loo, ev_keep, _, _ = loo_and_subset(tok, ev_rows, SEQ)
    print("train keep=%d (LOO=0:%d, LOO>=4:%d) | eval keep=%d"
          % (len(tr_keep), int((tr_loo == 0).sum()), int((tr_loo >= 4).sum()), len(ev_keep)))

    if a.arm == "precheck":
        plan = [("precheck", D_PRECHECK, s, lr)
                for lr in LR_PRECHECK for s in PRECHECK_SEEDS]
    else:
        plan = [("main", d, s, 0.003) for d in D_MAIN for s in SEEDS]
    if a.lrs:
        keep = {float(x) for x in a.lrs.split(",")}
        plan = [p for p in plan if p[3] in keep]
        assert plan, "--lrs %s 与该 arm 的计划无交集" % a.lrs

    have = done_keys()
    for arm, d, seed, lr in plan:
        if (arm, d, seed, lr) in have:
            print("跳过已完成 %s d=%d seed=%d lr=%s" % (arm, d, seed, lr))
            continue
        print("\n>>> %s d=%d seed=%d lr=%s" % (arm, d, seed, lr), flush=True)
        rec = run_one(arm, d, seed, lr, ck, tok, tr_rows, ev_rows, tr_pack, ev_pack,
                      tr_loo, tr_keep, ev_loo, ev_keep)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("    R1(LOO=0)=%.4f  R2(LOO>=4)=%.4f  trainNLL=%.4f  train1st=%.4f  "
              "Δtr=%.4f±%.4f  Δev=%.4f±%.4f  %.0fs"
              % (rec["r1_loo0_acc"], rec["r2_loo4_acc"], rec["train"]["answer_nll"],
                 rec["train"]["first_answer_token_accuracy"], rec["delta_train"],
                 rec["delta_train_se"], rec["delta_eval"], rec["delta_eval_se"],
                 rec["wall_sec"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())