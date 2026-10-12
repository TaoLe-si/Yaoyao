"""§69.52 **解耦 steps 与 N**：把步数钉死在 462 重扫 N，才能谈「数据量 vs train 最优」。

## 本脚本存在的唯一理由：§69.51 的自变量是共线的

§69.51（`exp_n_sweep.py`）用 `epochs ≡ 40`，于是
`steps = ceil(40·N/256) ≈ 0.15625·N` —— **steps 与 N 精确成比例，一个自由度都没有**。
这条扫描轴同时控制三件事：

| 轴 | 随 N 变的是什么 |
|---|---|
| 数据量 | 要记的**唯一行数** N |
| 优化预算 | 梯度更新**次数** steps（= 0.15625·N） |
| token 曝光 | 见到的 answer token **总量**（epochs × 语料 token 数 ∝ N） |

⇒ §69.51 读到的「train tok_acc 随 N **先升后降**、峰在 N=2,952」**不能归因给数据量**：
倒 U 的左半段完全可能是「N 太小、步数不够、还没训完」，
右半段才是「要记的太多、记不下」。两种解释在共线设计下**不可分辨**。

## 网格：两条线，交于一点

| 线 | N=738 | 1,476 | 2,952 | 5,904 | 11,808 |
|---|---|---|---|---|---|
| **epochs≡40**（§69.51 已有，勿重跑） | 116 | 231 | 462 | 923 | — |
| **steps≡462**（本脚本） | 新 | 新 | **复用 §69.51** | 新 | 新 |

★ N=2,952 @ 462 步是两条线的**交点**：配置逐字相同（同 seed 11 / 同 `tr[:2952]` /
同 `d=128` / 同 `n_slots=4096` / 同 `micro=64` / 同 462 步）⇒ 直接复用 §69.51 的读数，
**不重跑**（重跑只会把同一个数再算一遍）。脚本启动时会用 epochs 恒等式校验这份复用。

## 判据（先登记，每条都可失败）

| 编号 | 预测 | 被推翻意味着 |
|---|---|---|
| **D1** | steps≡462 上 tok_acc 随 N **单调降** | 不单调 ⇒ §69.51 峰值的右侧下降另有原因，首选嫌疑 `n_slots=4096` 装不下 N>4096 |
| **D2** | 若 D1 成立 ⇒ epochs≡40 的峰是**步数×数据量的合成产物**，不是数据量的最优点 | D1 不成立 ⇒ 合成轴解释作废，需另找机制 |
| **D3** | N/n_slots 过 1（N=5904→1.44、11808→2.88）后若**加速**下滑 ⇒ 支持槽位饱和解释 | 不加速 ⇒ 槽位饱和不是主因 |

判据可失败性的证明放在 `probe_n_curve_facts.py`（含人工反转序列的负向对照）。

## 追加单格：钉死断崖位置（`--ns`）

§69.52 跑出 steps≡462 线上 tok_acc 从 N=2952 的 0.9970 掉到 N=5904 的 0.7779。
断崖落在 **N=2952 与 N=5904 之间**，而 **n_slots=4096 正好在这个区间里**。

| 编号 | 预测（先登记） | 成立/被推翻 |
|---|---|---|
| **C1** | N=4096 @462 步的 tok_acc **也塌**（<0.9） | ⇒ 断崖就在 n_slots，容量单位是「槽」 |
| | N=4096 @462 步的 tok_acc **仍高**（≥0.99） | ⇒ 断崖在 2952~4096 之间，n_slots 不是那把尺子 |

★这条假设**不是**「N 超过槽数就装不下」：`build_engram_from_examples` 建的是 n-gram
计数直方图（`dialogue_small.py:351-357`），实测 alive_ratio 在 N=738 就是 99.88%、
一路到 11808 都是 100% ⇒ **活槽率是常量指标、没有区分度**（规则 9）。
真正随 N 变的是 token/槽：6.94 → 150.28（21.6 倍）。C1 检验的是
「槽数 4096 是不是那个硬边界」，不是「活槽率」。

用法：`python exp_n_steps_decoupled.py`（原 5 档）
      `python exp_n_steps_decoupled.py --ns 4096`（追加单格）
"""
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

OUT = os.path.join(HERE, "n_steps_decoupled.jsonl")
SWEEP = os.path.join(HERE, "n_sweep.jsonl")

D = 128
STEPS_FIXED = 462        # = §69.51 N=2952 那一格的步数（ceil(40*2952/256)=462）
BS = 256
LR = 0.003
SEED = 11
MICRO = 64
N_SLOTS = 4096
SEQ_LEN = 512            # = E2.SEQ，启动时会 assert 相等
NS = [738, 1476, 2952, 5904, 11808]
REUSE_N = 2952           # 两条线的交点 ⇒ 复用 §69.51，不重跑
HELD_PROBE = 2000        # held 教师强制探针行数（与 §69.51 相同口径）
EVAL_K = 200
EVAL_SEED = 12345


def epochs_of(n, steps):
    """恒等式：epochs = BS·steps/n。steps = ceil(40n/256) ⇒ epochs ∈ [40, 40+BS/n)。"""
    return BS * steps / float(n)


def epochs_bound(n):
    """steps=ceil(EPOCHS_TARGET·n/BS) 允许的 epochs 双侧界（规则 12：容差由被测量推出）。"""
    return (40.0, 40.0 + BS / float(n))


def teacher_acc_count(model, x, s, l):
    """教师强制口径的 answer token accuracy，**返回 (acc, 分母 token 数)**。

    ★为什么多返回分母：acc = 命中/位置，两个不同格的分母不同 ⇒ 分辨率不同。
      只报 acc 不报分母，下一个人无法判断两条读数的差是否落在同一把尺子的刻度内
      （1 次命中在 2000 token 上是 5e-4，在 200000 token 上是 5e-6）。
    全量、不抽样；口径与 §69.51 `exp_n_sweep.py:65-80` 逐字一致（否则两批读数不可配对）。
    """
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for i in torch.arange(x.shape[0]).split(32):
            bx = x[i].to(E2.DEV)
            sb = s[i].to(E2.DEV)
            lb = l[i].to(E2.DEV)
            pos = torch.arange(bx.shape[1], device=E2.DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            mask = valid[:, 1:] & (pos[:, 1:] >= sb.view(-1, 1))
            hit += int(((logits.argmax(-1) == bx[:, 1:].to(E2.DEV)) & mask).sum())
            tot += int(mask.sum())
    return hit / max(tot, 1), tot


def gen_em(model, tok, rows, k=EVAL_K, seed=EVAL_SEED):
    """生成式 EM + char-F1（抽 k 条，种子固定）。

    ★口径必须与 §69.51 一致才可配对：同样 k=200 / seed=12345 / 同样 `rnd.sample(rows,...)`、
      同样**逐行** `greedy_generate`（不用 `greedy_generate_batch`：批量与单行虽共用
      `_finalize_generation`，但前向的 batch 维度不同 ⇒ 浮点求和序不同，会让读数不可配对）。
    """
    rnd = random.Random(seed)
    rs = rnd.sample(rows, min(k, len(rows)))
    em = 0
    f1s = []
    for r in rs:
        out = E2.DS.greedy_generate(model, tok, r, max_new_tokens=140,
                                    device=E2.DEV, context_limit=512,
                                    temperature=0.0, utf8_safe=True)
        em += int(out["generated"].strip() == out["reference"].strip())
        f1s.append(E2.DS.char_f1(out["generated"], out["reference"]))
    return em, len(rs), (sum(f1s) / len(f1s) if f1s else 0.0)


def anchor_checks(rec):
    """复用 §69.51 那一格的「同口径」检验。抽成函数是为了能对它跑负向对照。"""
    lo, hi = epochs_bound(REUSE_N)
    e_calc = epochs_of(REUSE_N, STEPS_FIXED)
    # ★容差由**被测量的表示精度**推出，不是拍脑袋（规则 12）：
    #   `exp_n_sweep.py:133` 落盘写的是 `round(BS*steps/n, 3)` ⇒ 记录值与真值的偏差
    #   最多是最后一位的半个单位 = 5e-4。实测偏差 4.065e-5（真值 40.0650407，
    #   落盘 40.065）**正是这个量化**。我第一版写 1e-6，把检查判红了 ——
    #   那是「检查写错」不是「数据错」（规则 5）。
    tol_round = 5e-4
    return [
        ("steps 一致", rec["steps"] == STEPS_FIXED),
        ("epochs 算值在 ceil 双侧界内", lo <= e_calc < hi),
        ("落盘 epochs 与算值一致（容差 %.0e = round(...,3) 的量化半宽）" % tol_round,
         abs(rec["epochs"] - e_calc) < tol_round),
        ("d 一致", rec["d"] == D),
    ]


def anchor_check():
    """锚点自检（规则 11）：复用的 §69.51 那一格必须与本脚本**同口径**。

    ★负向对照（规则 9/29）：把 epochs 改到量化半宽之外，判据必须拒绝；
      并且先断言「判据的输入真的变了」，否则这道对照是形式主义。
    """
    if not os.path.exists(SWEEP):
        raise SystemExit("缺 %s，无法做锚点自检" % os.path.basename(SWEEP))
    recs = [json.loads(l) for l in open(SWEEP, encoding="utf-8") if l.strip()]
    hit = [r for r in recs if r["n"] == REUSE_N]
    if len(hit) != 1:
        raise SystemExit("§69.51 里 N=%d 的格数 = %d（应为 1）" % (REUSE_N, len(hit)))
    r = hit[0]
    checks = anchor_checks(r)

    bad = dict(r)
    bad["epochs"] = r["epochs"] + 0.01
    assert abs(bad["epochs"] - r["epochs"]) > 5e-4, \
        "负向对照没把值改到量化半宽之外 ⇒ 对照无效"
    bad_checks = anchor_checks(bad)
    if all(ok for _n, ok in bad_checks):
        raise SystemExit("锚点判据恒真（伪造 epochs 仍全 OK）⇒ 拒绝开跑")

    print("【锚点自检：复用 §69.51 的 N=%d @ %d 步】" % (REUSE_N, STEPS_FIXED))
    for name, ok in checks:
        print("  %-52s %s" % (name, "OK" if ok else "FAIL"))
        if not ok:
            raise SystemExit("锚点自检失败 ⇒ 复用不成立，拒绝开跑（规则 11）")
    lo, hi = epochs_bound(REUSE_N)
    print("  epochs = BS*steps/N = %.7f，界 [%.1f, %.1f)，落盘 %.3f，量化偏差 %.2e"
          % (epochs_of(REUSE_N, STEPS_FIXED), lo, hi, r["epochs"],
             abs(r["epochs"] - epochs_of(REUSE_N, STEPS_FIXED))))
    print("  负向对照：epochs +0.01（超出量化半宽）=> 判定 %s ⇒ 判据可失败"
          % ("全 OK" if all(ok for _n, ok in bad_checks) else "拒绝（符合预期）"))
    return r


def main():
    # ★--ns 用来追加单格做机制验证（比如钉死断崖位置）。默认仍跑原 5 档。
    #   落盘键含全部自变量，续跑口径不会把追加的格当成重复。
    global NS
    argv = [a for a in sys.argv[1:]]
    save_path = None
    if "--save" in argv:
        i = argv.index("--save")
        save_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if "--ns" in argv:
        i = argv.index("--ns")
        NS = [int(v) for v in argv[i + 1].split(",") if v.strip()]
        argv = argv[:i] + argv[i + 2:]
        if argv:
            raise SystemExit("未识别的参数：%s" % " ".join(argv))
    print("=" * 78)
    print("§69.52 解耦扫描：steps 钉死 %d，重扫 N" % STEPS_FIXED)
    print("=" * 78)
    print("  d=%d  n_slots=%d  bs=%d  lr=%g  seed=%d  micro=%d"
          % (D, N_SLOTS, BS, LR, SEED, MICRO))
    print("  N 档：%s（%d 复用 §69.51）" % (NS, REUSE_N))
    print("  每格 epochs = BS*steps/N = %s"
          % ["%.1f" % epochs_of(n, STEPS_FIXED) for n in NS])
    anchor = anchor_check()

    import model as M

    # ── 结构自检：cfg 派生量在造模型**之前**先算（§1A 的 D/E 步） ──────
    assert E2.BS == BS, "E2.BS=%d != 本脚本 BS=%d ⇒ 口径不一致" % (E2.BS, BS)
    assert E2.LR == LR, "E2.LR=%g != %g ⇒ train_steps 用的是模块全局，别被打印值骗" % (E2.LR, LR)
    assert E2.SEED == SEED, "E2.SEED=%d != %d" % (E2.SEED, SEED)
    assert E2.SEQ == SEQ_LEN, "E2.SEQ=%d != %d" % (E2.SEQ, SEQ_LEN)
    assert E2.HEAD_DIM == 16, "E2.HEAD_DIM=%d ⇒ head_dim 口径变了" % E2.HEAD_DIM
    n_heads = D // E2.HEAD_DIM
    print("  形状推导：d=%d / HEAD_DIM=%d => n_heads=%d；w/o 形状 [%d, %d]"
          % (D, E2.HEAD_DIM, n_heads, n_heads, E2.HEAD_DIM))
    assert n_heads * E2.HEAD_DIM == D, "d 必须被 HEAD_DIM 整除"
    # §69.51 同为 MICRO=64 且 n_heads = 128//16 = 8 ⇒ 同样是 512。
    # （别写死 64*4：那是修 bug 前的 n_heads=4，见 run_qa_tables.py 那次形状不匹配事故。）
    assert MICRO * n_heads == 64 * (128 // 16), "与 §69.51 的 micro×n_heads 不一致"
    print("  micro×n_heads = %d（§69.51 同为 64×8 = %d，口径一致）"
          % (MICRO * n_heads, 64 * (128 // 16)))

    E2.check_free_memory()
    E2.load_all()
    rows = E2.load_rows()
    tr, hd = E2.split_rows(rows)
    print("  全量 train %d / held %d；held 探针取前 %d 行（与 §69.51 同口径）"
          % (len(tr), len(hd), HELD_PROBE))
    hx, hs, hl = E2.encode_cached(hd[:HELD_PROBE], E2._TOK, E2.SEQ)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["n"], r["steps"], r["d"], r["n_slots"], r["seed"],
                          r["micro"], r["held_probe"]))
    print("  续跑口径：已落盘 %d 格，键 = (n, steps, d, n_slots, seed, micro, held_probe)"
          % len(done))

    for n in NS:
        # ★要存权重就必须**真训**这一格：复用 §69.51 的读数可以，但读数里没有 state_dict。
        if n == REUSE_N and save_path is None:
            print()
            print("  N=%-6d **复用 §69.51**（两条线交点，不重跑）tok_acc=%.4f EM=%d/%d"
                  % (n, anchor["train_tok_acc"], anchor["train_em"], anchor["train_em_n"]),
                  flush=True)
            continue
        key = (n, STEPS_FIXED, D, N_SLOTS, SEED, MICRO, HELD_PROBE)
        if key in done and save_path is None:
            print("  N=%-6d 已落盘，跳过" % n, flush=True)
            continue
        if key in done:
            print("  N=%-6d 已落盘，但 --save 需要权重 => 重训这一格（多花 ~%d s）"
                  % (n, int(STEPS_FIXED * 0.45)), flush=True)
        t0 = time.time()
        sub = tr[:n]
        x, s, l = E2.encode_cached(sub, E2._TOK, E2.SEQ)
        cfg = dict(E2._CK["cfg"])
        cfg["d"], cfg["n_heads"], cfg["n_slots"] = D, n_heads, N_SLOTS
        cfg["pos_abs_max"], cfg["use_qpool"] = 512, False
        torch.manual_seed(SEED)
        m = M.EngramRuleLM(M.RCfg(**cfg)).to(E2.DEV)
        # ★记 alive：表不是「一行一槽」，而是 n-gram 计数直方图
        #   （dialogue_small.py:351-357：counts[slot_id] += 1，alive = counts>=min_count）
        #   ⇒ N 变大不会「行装不下」，而是 **distinct n-gram 数逼近 n_slots 发生别名碰撞**。
        #   alive_slots 直接量出这个饱和度，是解释曲线为何转向的**便宜读数**（不需训练）。
        binf = E2.DS.build_engram_from_examples(m, x, l)
        trace, rinfo = E2.train_steps(m, x, s, l, STEPS_FIXED, MICRO, 0.0,
                                      resume=None)
        assert isinstance(trace, list) and isinstance(rinfo, dict), \
            "train_steps 返回 (%s, %s)，不是 (trace, rinfo)" % (type(trace).__name__,
                                                                type(rinfo).__name__)
        acc, ntok = teacher_acc_count(m, x, s, l)
        em_tr, n_tr, f1_tr = gen_em(m, E2._TOK, sub)
        hacc, hntok = teacher_acc_count(m, hx, hs, hl)
        wall = time.time() - t0
        rec = dict(n=n, d=D, n_slots=N_SLOTS, seed=SEED, micro=MICRO,
                   held_probe=HELD_PROBE, steps=STEPS_FIXED,
                   epochs=round(epochs_of(n, STEPS_FIXED), 3),
                   train_tok_acc=acc, train_answer_tokens=ntok,
                   train_em=em_tr, train_em_n=n_tr, train_f1=f1_tr,
                   held_tok_acc=hacc, held_answer_tokens=hntok,
                   alive_slots=int(binf["alive_slots"]),
                   alive_ratio=round(float(binf["alive_ratio"]), 6),
                   trace_tail=trace[-2:] if trace else [], wall_sec=round(wall, 1),
                   sec_per_step=round(wall / STEPS_FIXED, 4))
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # ★存盘必须在 `del m` **之前**完成（否则取不到 state_dict）。
        #   格式与 `exp_scaling_grid_v2.py:743-744` 的成品完全一致：
        #   cfg 取 `model.c.__dict__`（RCfg 是 dataclass，带默认值的字段也在里面），
        #   少写一个字段，下一个人重建模型就会用到不同默认值 —— 那正是 §69.35 栽过的坑。
        if save_path is None:
            path = os.path.join(
                HERE, "engram_rule_nsweep_n%d_d%d_s%d_wd0p0.pt" % (n, D, STEPS_FIXED))
        else:
            path = os.path.join(HERE, save_path)
        cell = dict(n=n, d=D, steps=STEPS_FIXED, weight_decay=0.0, lr=LR,
                    bs=BS, seed=SEED, head_dim=E2.HEAD_DIM, micro_batch=MICRO,
                    seq=E2.SEQ, epochs=round(epochs_of(n, STEPS_FIXED), 3),
                    gate=E2.GATE, source="exp_n_steps_decoupled.py",
                    train_tok_acc=acc, train_em="%d/%d" % (em_tr, n_tr),
                    held_tok_acc=hacc)
        tmp = path + ".tmp"
        torch.save(dict(cfg=m.c.__dict__, state_dict=m.state_dict(),
                        tokenizer=E2._TOK_META, cell=cell), tmp)
        os.replace(tmp, path)          # 原子替换：中途被杀不会留下半个权重
        print("  已存盘 %s（%.1f MB）" % (os.path.basename(path),
                                       os.path.getsize(path) / 1048576.0), flush=True)
        print()
        print("  N=%-6d steps=%-4d epochs=%-6.1f tok_acc=%.4f (分母 %d)  "
              "EM=%d/%d  F1=%.4f  held=%.4f  alive=%d/%d  %.0fs (%.3f s/步)"
              % (n, STEPS_FIXED, rec["epochs"], acc, ntok, em_tr, n_tr, f1_tr,
                 hacc, rec["alive_slots"], N_SLOTS, wall, rec["sec_per_step"]),
              flush=True)
        del m
        if E2.DEV == "cuda":
            torch.cuda.empty_cache()

    print()
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())