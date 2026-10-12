# -*- coding: utf-8 -*-
"""§69.35 用**当前落点**（N=124593 × d=128 训练中途）的权重做**抽样 200** 问答。

── 桃的要求 ────────────────────────────────────────────────────────
「测试当前存档的权重完整训练集问答，和留出集问答」→ 我第一版拿**旧的 v11 出货权重**跑，
被指出测错了对象。要测的是 `N=124593 × d=128` 那格**正在训、还没训完**的权重。
「当前落点 NLL 是多少，抽样测试，200 条」→ 每侧 **200 条**抽样，报落点 NLL。

── 权重从哪来 ──────────────────────────────────────────────────────
`resume_v2/resume_v2_n124593_d128_wd0p0_s19468_mb64_slots4096.pt` 的 `["model"]`。
那里面有**训练中途**的权重 + optimizer + scheduler + RNG + trace。
**只读**：不训练、不续训、不覆盖恢复点。

── 口径 ────────────────────────────────────────────────────────────
1. **抽样 200 条/侧**（桃本轮指定）。判据 G1 查落盘行数必须 == 200。
   ★抽样 ⇒ **无全量结论**，且**单 seed 无种子极差**（AGENTS 规则 16）⇒ 不报置信区间。
2. cfg 从**恢复键**读，不猜：d / n_slots / seed / lr / seq / bs / micro_batch / steps。
3. **Engram 表仍要建**，且用该格**自己的 train 集**（124,593 行）——
   §69.28.1 实测：不建表 ⇒ `alive` 全 False ⇒ `model.py:497` 的 `o = o*keep`
   把注意力输出全乘 0 ⇒ 退化成 emb+FFN，测的不是这个模型。

── 判据（可失败）───────────────────────────────────────────────────
| 编号 | 判据 | 失败意味着 |
|---|---|---|
| **G1** | 每侧落盘行数 == 200 | 抽样数不对 |
| **G2** | alive 槽位数 > 0 | 忘了建表，测的是残壳 |
| **G3** | train / held 抽样的**问题文本**集合不相交 | 泄漏 |
| **G4** | 落点 NLL 与恢复点 trace 末点同量级 | 两条口径差太多 ⇒ 有一条算错 |

★G4 的口径说明：**落点 NLL 是教师强制的 answer NLL**（`answer_metrics`，
  与网格 `probe.probe_nll` 同口径、在 held 探针集上算），
  **不是** trace 里的 train loss（那是训练中随机批的 answer loss）。
  两者同量级（差 < 0.5）才算自洽；差太多说明有一条算错。

用法：`python run_qa_tables.py`（200+200 行，几分钟）
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

RP = os.path.join(HERE, "resume_v2",
                  "resume_v2_n124593_d128_wd0p0_s19468_mb64_slots4096.pt")
# ★权重来源（2026-10-12 改两次）：
#   ① 先改成满步成品 `engram_rule_scaling_v2_n124593_d128_wd0p0.pt`
#      （上一版用恢复点，恢复点不存 cfg，我自己重建 ⇒ 栽在 n_heads 上）。
#   ② 现改由 `--ckpt` 指定，默认仍是满步成品。这样 500 步续训权重
#      （`engram_rule_extra500_lr3e-05.pt`）能用**完全同一套脚本、同一抽样种子**
#      跑出可对拍的读数，而不是另写一份脚本（另写就换了尺子）。
CKPT = os.path.join(HERE, "engram_rule_scaling_v2_n124593_d128_wd0p0.pt")
OUT_TRAIN = os.path.join(HERE, "qa_train_s200.jsonl")
OUT_HELD = os.path.join(HERE, "qa_held_s200.jsonl")
OUT_SUM = os.path.join(HERE, "qa_summary.json")
SAMPLE = 200
CALIB = 5                  # ★先跑 5 条看单行耗时，再放开到 SAMPLE
MAX_NEW = 140
CTX_LIMIT = 512
POS_ABS_MAX = 512
SAMPLE_SEED = 12345          # 固定抽样种子 ⇒ 可复现（单 seed，无种子极差）


def load_cell_model():
    """从恢复点读出训练中途的权重。**只读，不写恢复点、不训练。**

    ★★AGENTS 规则 7（这次真的栽了，记下来）：我第一版**自己拼 cfg**，
      按出货 v11 的口径把 `n_heads` 写死 4 ⇒ `head_dim = 128/4 = 32` ⇒
      `model.py:316-319` 的 `wg/wq/wu` 变成 [16,32]、`wd` 变成 [32,16]，
      而 d=128 的网格权重是 `n_heads=8` ⇒ `head_dim=16` ⇒ [16,16]
      ⇒ `load_state_dict` 形状不符、脚本**报错退出**。
      我却在汇报里说"200 条慢"，把**崩溃**当成了**慢** —— 这类错误最贵。

    ⇒ 修法：cfg 用 `E2.make_model` 的派生口径（`n_heads = d // HEAD_DIM`，
      HEAD_DIM=16 恒定）。恢复点里**没存 cfg**（只有恢复键），所以 d / n_slots /
      seed 全从恢复键取。
      ★但 `make_model` 内部会调 `DS.build_engram_from_examples(model, _TRAIN_X, _TRAIN_L)`，
        而 `_TRAIN_X` 是 `run_cell` 里设的**模块全局**，单独跑本脚本时不存在
        ⇒ NameError。所以这里**不调 `make_model`**，照它的口径自己构，
        但 `n_heads` 那一步**逐字照抄** `d // E2.HEAD_DIM`，不自己拍。
    """
    import model as M                      # ★规则 7：E2 没有导出 `M`
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])                  # ★逐字用成品自带的 cfg，一个字段都不改
    cell = dict(ck.get("cell") or {})
    torch.manual_seed(E2.SEED)             # 初始化用网格口径的 seed
    m = M.EngramRuleLM(M.RCfg(**cfg)).to(E2.DEV)
    # ★形状自检：在 load_state_dict **之前**逐项比，报错里直接指出是哪一维
    #   （AGENTS §1A 步 E —— 不要等 load_state_dict 抛一堆看不懂的行）
    sd = ck["state_dict"]
    want = {n: tuple(p.shape) for n, p in m.named_parameters()}
    bad = [(n, tuple(sd[n].shape), want[n]) for n in sd if n in want
           and tuple(sd[n].shape) != want[n]]
    assert not bad, "cfg 与权重形状不符：%s" % bad[:4]
    m.load_state_dict(sd)                  # ★strict=True：键/形状不符必须炸
    m.eval()
    return m, cfg, cell, []


def gen_split(model, tok, rows, tag, path=None):
    """逐行生成一份问答表。返回 (行数, em 数, char_f1 和, 用时)。

    ★`path` 由调用方传入（默认按 tag 推导）：这样 `--tag` 能把两次不同权重的
      读数**分开落盘**，不会互相覆盖 —— 否则对拍时读到的还是旧权重那版。
    """
    n = len(rows)
    t0 = time.time()
    n_em, sum_f1 = 0, 0.0
    path = path or (OUT_TRAIN if tag == "train" else OUT_HELD)
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            out = E2.DS.greedy_generate(model, tok, row, max_new_tokens=MAX_NEW,
                                        device=E2.DEV, context_limit=CTX_LIMIT,
                                        temperature=0.0, utf8_safe=True)
            g, r = out["generated"], out["reference"]
            em = int(g.strip() == r.strip())
            f1 = E2.DS.char_f1(g, r)
            n_em += em
            sum_f1 += f1
            f.write(json.dumps(dict(
                idx=i, question=row["question"], reference=r, generated=g,
                em=em, char_f1=f1, generated_tokens=out["generated_tokens"],
                eos=out["eos"], truncated_context=out["truncated_context"],
                repeated_token_rate=out["repeated_token_rate"],
            ), ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0 or i + 1 == n:
                print("  [%s] %d / %d  EM=%d (%.3f)  F1=%.4f  已用 %.0f s"
                      % (tag, i + 1, n, n_em, n_em / (i + 1),
                         sum_f1 / (i + 1), time.time() - t0), flush=True)
        f.flush()                          # ★关文件前必须 flush，否则 G1 数行数会读到 0
        os.fsync(f.fileno())               # ★连 OS 缓冲一起刷，避免读到陈旧内容
    disk = sum(1 for _ in open(path, encoding="utf-8"))
    assert disk == n, "G1a 失败：%s 落盘 %d 行 ≠ 生成的 %d" % (tag, disk, n)
    return n, n_em, sum_f1, time.time() - t0


def main():
    global CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT, help="权重来源（默认满步成品）")
    ap.add_argument("--tag", default=None, help="输出文件名后缀，避免覆盖上一版读数")
    ap.add_argument("--n", type=int, default=0,
                    help="★该权重实际训练过的 train 行数。0=用全部 train。\n"
                         "  必须传对：建 Engram 表与 train 侧抽样**都**只用这前 N 行，\n"
                         "  否则 alive 掩码和模型训练时不是同一个（§69.52.9）。")
    ap.add_argument("--probe", type=int, default=0,
                    help="held 教师强制探针的行数；0=全量。\n"
                         "  传 2000 可与 §69.51/§69.52 的 held_tok_acc 同口径对拍。")
    a = ap.parse_args()
    CKPT = a.ckpt
    out_tr = OUT_TRAIN if not a.tag else OUT_TRAIN.replace(".jsonl", "_%s.jsonl" % a.tag)
    out_hd = OUT_HELD if not a.tag else OUT_HELD.replace(".jsonl", "_%s.jsonl" % a.tag)
    out_sum = OUT_SUM if not a.tag else OUT_SUM.replace(".json", "_%s.json" % a.tag)
    print("=" * 78)
    print("§69.35 抽样 %d 问答（权重 = %s）" % (SAMPLE, os.path.basename(CKPT)))
    print("=" * 78)
    E2.load_all()
    rows = E2.load_rows()
    train_rows, held_rows = E2.split_rows(rows)
    # ★★口径闸：该权重只训练过前 N 行 ⇒ 建表与 train 抽样都必须截到这 N 行。
    #   原版用**全部 124,593 行**建 Engram 表，而 `build_engram_from_examples` 是
    #   `counts[slot] += 1` 后 `alive = counts >= min_count`（cfg 里 min_count=1）
    #   ⇒ 行数越多活槽越多，`model.py:497` 的 `o = o*keep` 掩码就变，
    #   **测出来的根本不是这一格权重**。min_count=1 时 alive 恰是「该槽被用过」。
    n_train = a.n if a.n else len(train_rows)
    if a.n and a.n > len(train_rows):
        raise SystemExit("--n=%d 超过 train 行数 %d" % (a.n, len(train_rows)))
    if a.n:
        train_rows = train_rows[:a.n]
        print("  ★train 侧截到前 %d 行（该权重的实际训练集）" % len(train_rows))
    print("  语料 %d 行 → train %d / held %d" % (len(rows), len(train_rows), len(held_rows)))

    model, cfg, cell, _ = load_cell_model()
    print()
    print("  权重来源：%s" % os.path.basename(CKPT))
    print("  cell：%s" % json.dumps(cell, ensure_ascii=False, default=str))

    # ★Engram 表：用该格**自己的** train 集建（G2），即上面截断后的 train_rows
    x, s, l = E2.encode_cached(train_rows, E2._TOK, E2.SEQ)
    info = E2.DS.build_engram_from_examples(model, x, l)
    print("  建 Engram（%d 行）：alive %d / %d (%.4f)"
          % (len(train_rows), info["alive_slots"], info["n_slots"], info["alive_ratio"]))
    assert info["alive_slots"] > 0, "G2 失败：alive 仍全 False ⇒ 测的是残壳（§69.28.1）"
    print("  ⇒ G2 成立")

    # ★落点 NLL：held 探针集上的教师强制 answer NLL，与网格 probe 同口径
    probe_rows = held_rows[:a.probe] if a.probe else held_rows
    px, ps, pl = E2.encode_cached(probe_rows, E2._TOK, E2.SEQ)
    am = E2.DS.answer_metrics(model, px, ps, pl, 32, E2.DEV)
    print()
    print("  ★落点 NLL（held 前 %d 行教师强制，与网格 probe 同口径）：" % len(probe_rows))
    print("    answer_nll      = %.6f" % am["answer_nll"])
    print("    answer_ppl      = %.4f" % am["answer_ppl"])
    print("    token_accuracy  = %.6f" % am["answer_token_accuracy"])
    print("    first_tok_acc   = %.6f" % am["first_answer_token_accuracy"])
    print("    answer_tokens   = %s" % "{:,}".format(am["answer_tokens"]))

    # ── 抽样 200 ───────────────────────────────────────────────────
    g = torch.Generator().manual_seed(SAMPLE_SEED)
    idx_tr = torch.randperm(len(train_rows), generator=g)[:SAMPLE].tolist()
    idx_hd = torch.randperm(len(held_rows), generator=g)[:SAMPLE].tolist()
    samp_tr = [train_rows[i] for i in idx_tr]
    samp_hd = [held_rows[i] for i in idx_hd]

    # ★G3：两侧抽样的问题文本必须不相交
    q_tr = {r["question"].strip() for r in samp_tr}
    q_hd = {r["question"].strip() for r in samp_hd}
    inter = q_tr & q_hd
    assert not inter, "G3 失败：两侧抽样有 %d 个问题交集" % len(inter)
    print("  ⇒ G3 成立（train 200 个问题 × held 200 个问题，交集 0）")

    print()
    print("生成参数：greedy / max_new_tokens=%d / context_limit=%d" % (MAX_NEW, CTX_LIMIT))
    print()
    # ★先跑 CALIB 条看单行耗时，再放开到 SAMPLE。
    #   上一轮我把「报错退出」当成了「200 条很慢」，白等一轮。先校准速度，
    #   不让任何一次 200 条的跑动建立在对耗时的猜测上。
    g_cal = torch.Generator().manual_seed(SAMPLE_SEED)
    cal_rows = [train_rows[i] for i in torch.randperm(len(train_rows),
                                                      generator=g_cal)[:CALIB].tolist()]
    t_cal = time.time()
    for row in cal_rows:
        E2.DS.greedy_generate(model, E2._TOK, row, max_new_tokens=MAX_NEW,
                              device=E2.DEV, context_limit=CTX_LIMIT,
                              temperature=0.0, utf8_safe=True)
    per = (time.time() - t_cal) / CALIB
    print("  速度校准：%d 条 / %.1f s ⇒ %.2f s/行；%d 条预计 %.0f s"
          % (CALIB, time.time() - t_cal, per, SAMPLE, per * SAMPLE))
    print()
    print("── TRAIN 侧抽样 %d（训练见过的问题，只证明记住了）──" % SAMPLE)
    n_tr, em_tr, f1_tr, w_tr = gen_split(model, E2._TOK, samp_tr, "train", out_tr)
    print()
    print("── HELD 侧抽样 %d（完全没见过的问题，才证明泛化）──" % SAMPLE)
    n_hd, em_hd, f1_hd, w_hd = gen_split(model, E2._TOK, samp_hd, "held", out_hd)

    # ── G1 行数 ────────────────────────────────────────────────────
    # ★必须查**这次实际写出的** out_tr / out_hd。原版查的是 OUT_TRAIN / OUT_HELD
    #   （模块默认名），于是加了 --tag 之后 G1 数的是**上一版**的文件，
    #   行数照样等于 200 ⇒ 判据恒真，形同虚设（这正是「元类判据拿自己赋的值比」的坑）。
    for tag, path in (("train", out_tr), ("held", out_hd)):
        disk = sum(1 for _ in open(path, encoding="utf-8"))
        assert disk == SAMPLE, "G1 失败：%s 落盘 %d 行 ≠ %d（%s）" % (tag, disk, SAMPLE,
                                                                    os.path.basename(path))
    print()
    print("  ⇒ G1 成立（%s / %s 各落盘 %d 行）"
          % (os.path.basename(out_tr), os.path.basename(out_hd), SAMPLE))

    summary = dict(
        weight_source=os.path.basename(CKPT) + "（只读）",
        cell=cell,
        train_rows_used=len(train_rows),
        probe_rows_used=len(probe_rows),
        alive=dict(slots=info["alive_slots"], n_slots=info["n_slots"],
                   ratio=info["alive_ratio"]),
        probe_teacher_forced=am,
        gen=dict(max_new_tokens=MAX_NEW, context_limit=CTX_LIMIT,
                 temperature=0.0, greedy=True, utf8_safe=True),
        sample=dict(n_per_side=SAMPLE, seed=SAMPLE_SEED),
        train=dict(rows=n_tr, em=em_tr, em_rate=em_tr / n_tr,
                   char_f1=f1_tr / n_tr, wall_sec=round(w_tr, 1)),
        held=dict(rows=n_hd, em=em_hd, em_rate=em_hd / n_hd,
                  char_f1=f1_hd / n_hd, wall_sec=round(w_hd, 1)),
        files=dict(train=os.path.basename(out_tr), held=os.path.basename(out_hd)),
    )
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print()
    print("=" * 78)
    print("【读数】权重 = %s（cell: n=%s d=%s steps=%s）"
          % (os.path.basename(CKPT), cell.get("n"), cell.get("d"), cell.get("steps")))
    print("=" * 78)
    print("  落点 held answer NLL  = %.6f   ppl = %.4f" % (am["answer_nll"], am["answer_ppl"]))
    print("  held token acc        = %.6f" % am["answer_token_accuracy"])
    print()
    print("  %-8s %7s %6s %9s %9s" % ("侧", "行数", "EM", "EM 率", "char-F1"))
    print("  %-8s %7d %6d %8.2f%% %9.4f" % ("train", n_tr, em_tr, 100 * em_tr / n_tr, f1_tr / n_tr))
    print("  %-8s %7d %6d %8.2f%% %9.4f" % ("held", n_hd, em_hd, 100 * em_hd / n_hd, f1_hd / n_hd))
    print()
    print("  ★抽样 %d/侧、单 seed ⇒ **无种子极差**，不报置信区间（规则 16）。" % SAMPLE)
    print("  ★train 高分只证明记住了；held 高分才证明泛化。")
    print("  产物：%s / %s / %s"
          % (os.path.basename(out_tr), os.path.basename(out_hd),
             os.path.basename(out_sum)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
