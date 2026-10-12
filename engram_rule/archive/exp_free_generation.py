"""★实验 B: 「答非所问」的判决 —— 教师强制 vs 自由生成, 以及**容量治不治得了它**。

**这是跑实验的脚本, 只产出原始读数**; 判决全部在 `probe_free_generation.py` 里,
按 DESIGN §56 的预登记规则执行。两者分开是为了不让"跑"的人顺手改判据。

────────────────────────────────────────────────────────────────────────
**这个实验要解决的矛盾**(DESIGN §48 之后一直没测过的那一格):

  · train 侧教师强制在 d=256 已经近乎完美(NLL 0.012 / R1 1.0000 / token acc ~1.0)
  · 但 train 侧**自由生成**在 d=64 只有 2/12 EM

  这两件事**不可能同时成立**, 除非: teacher-forcing 与 free-running **不是同一条路径**,
  或者"教师强制完美"这个读数并不是它看上去的那个意思。

────────────────────────────────────────────────────────────────────────
**口径纪律**

  · prompt/context: 用 checkpoint 真实 `sequence_length`(512), **不是**
    `evaluate_generations` 的默认 96 —— §50 已经在这上面栽过一次
    (`probe_rerank_mbr.py` 的 context_limit bug, 204 行里有 2 行 prompt > 96)。
  · 数据与 §48 完全同源(`split_records(seed=29)`), 并对 eval 侧逐元素断言。
  · 生成参数与 `train_qa_report.md` 的 12 条自由生成**同口径**:
    `temperature=0, repetition_penalty=1.0, no_repeat_ngram=0, utf8_safe=True, max_new_tokens=90`。
  · 逐 token 的答案 id 由**出货的** `greedy_generate` / `greedy_generate_batch` 提供
    (本轮给 `_finalize_generation` 加了只增不改的 `answer_ids` 字段), **不在本脚本里
    重写生成逻辑** —— 重写就等于换了一把尺子。

────────────────────────────────────────────────────────────────────────
**只追加, 不覆盖**: 每次 part 一行 JSON 写进 `free_generation_v1.jsonl`。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

# ★★★ 必须在 `import torch` **之前**设: 分配器在 CUDA 上下文创建时读一次。
# 本轮审计(§57)的实测: 默认缓存分配器在这个负载下碎片化得极厉害 ——
# 真正存活的张量只有 ~100 MiB, 而 `nvidia-smi` 看到的 reserved 一路涨:
#   d=64  mb=64 : 2274 → 3540 MiB (2 步)
#   d=128 mb=32 : 2632 → 3814 MiB (2 步)
#   d=256 mb=32 : 2596 → 3184 MiB, 把 8188 MiB 的卡占满
# 开 `expandable_segments` 后 reserved 变平且不涨:
#   d=64  mb=64 : 2150 MiB, 40 步不动
#   d=128 mb=32 : 2068 MiB, 30 步不动
# ★注意减 micro_batch **没用**(d=256 mb=32→16 反而 3184→3384): 涨的是保留池
#   而不是瞬时峰值, 所以只能治分配器, 不能治张量大小。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
OUT = os.path.join(HERE, "free_generation_v1.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
EVAL_BS = 32
N_ROWS = 204                       # 与 §50/§51 的 204 行口径一致
MAX_NEW = 90                       # 与 train_qa_report.md 的 12 条同口径
CONTEXT_LIMIT = 512                # checkpoint 真实 sequence_length, 不是默认值 96
HEAD_DIM = 16                      # 与 §48 一致: head_dim 恒 16, n_heads 随 d 变


def _gate_tag(names=("exp_free_generation.py", "check_free_generation_probe.py")):
    """★ 门禁指纹 = 测量脚本 + 检查脚本的 sha256。

    动机: 本轮踩过一次 —— 检查 17/17 通过之后我又改了测量代码(EOS 对齐/part1/
    误删又补回 main), **没重跑检查**就直接跑了实验, 产出的读数没有任何检查背书。
    指纹让这种数据**自动作废**: 改了任一文件, 旧记录的 gate 就对不上,
    `probe_free_generation.py` 会把它们排除而不是拿它们下结论。
    """
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in names:
        with open(os.path.join(here, name), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:12]


GATE = _gate_tag()


# --------------------------------------------------------------------- 数据

def load_data():
    """与 `exp_capacity_scaling.load_data` **逐行相同**(口径不一致 ⇒ 读数不可比)。"""
    import exp_capacity_scaling as exp
    ck, tok, tr_rows, ev_rows, tr_pack, ev_pack = exp.load_data()
    ctx = int(ck["cfg"]["pos_abs_max"])
    assert ctx == CONTEXT_LIMIT, "context_limit 应等于 checkpoint 的 pos_abs_max=%d" % ctx
    return ck, tok, tr_rows, ev_rows, tr_pack, ev_pack


def gold_answer_ids(x, starts, lengths, i):
    """第 i 行的 gold 答案 token(含末尾 [EOS]) —— 与 format_record 的切法一致。"""
    return x[i, int(starts[i]):int(lengths[i])].tolist()


def lcp_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# --------------------------------------------------------------------- 教师强制

@torch.no_grad()
def teacher_forced_rows(model, rows, x, starts, lengths, batch_size):
    """逐行答案段的 argmax / top1-top2 margin。

    返回值 pred[i][j] 是"喂 gold 前 j 个答案 token 后, 第 j 个答案位置的 argmax"。
    ★与 free-running 的第 j 步**输入完全相同**(gold 前缀), 所以两者的分歧位置
      在逻辑上必须相等 —— 判据 L1 就量这个。
    """
    out = []
    for bi in torch.arange(x.shape[0]).split(batch_size):
        xb = x[bi].to(DEV)
        logits = model(xb).float()
        for j in range(xb.shape[0]):
            s, l = int(starts[bi][j]), int(lengths[bi][j])
            lg = logits[j, s - 1:l - 1]            # 位置 t 的 logits 预测 token t+1
            pred = lg.argmax(-1)
            if lg.shape[1] >= 2:
                top2 = lg.topk(2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1])
            else:
                margin = torch.full((lg.shape[1],), float("inf"), device=DEV)
            out.append(dict(pred=pred.tolist(), margin=margin.tolist()))
    return out


def utf8_block_at_first_div(tok, mask_cache, byte_map, eos_id, gold, first_div):
    """free-running 在首个分歧位是否会因 UTF-8 允许集而选不出 gold token。

    ★这是 teacher-forcing 与 free-running 之间**唯一**的非对称: 生成路径会
      `masked_fill(~allowed, -inf)`, 而 `answer_metrics` 的 argmax 不会。
    返回 (是否被掩掉, 该位 allowed 集合大小)。
    """
    from dialogue_small import valid_token_bytes
    state, tail = (0, 0x80, 0xBF), b""
    # ★只推进 first_div-1 个 token: 要判断的是"第 first_div 位**之前**的状态下",
    #   gold[first_div-1] 允不允许被选。推进到 first_div 就是问"吃掉它之后的状态",
    #   那是**下一个**位置的状态 —— 第一版就是这么写错的(检查 P5a 抓到)。
    for k in range(first_div - 1):
        t = gold[k]
        if t != eos_id:
            state = valid_token_bytes(state, byte_map[t])
            tail = (tail + byte_map[t])[-2:]
    allowed = mask_cache.allowed_mask(state, tail)
    n_allowed = int(allowed.sum())
    if first_div - 1 >= len(gold):
        return False, n_allowed
    return (not bool(allowed[gold[first_div - 1]])), n_allowed


# --------------------------------------------------------------------- 自由生成

def free_running(model, tok, rows, max_new_tokens=MAX_NEW, context_limit=CONTEXT_LIMIT,
                 temperature=0.0, repetition_penalty=1.0, no_repeat_ngram=0,
                 utf8_safe=True, seed=0, batch_min_size=2):
    """走**出货**的 `evaluate_generations`(它转调 greedy_generate / _batch),
    取回每行的 `answer_ids` 与解码文本。本脚本不复制任何生成逻辑。"""
    from dialogue_small import evaluate_generations, normalize_text, char_f1
    metrics, outputs = evaluate_generations(
        model, tok, rows, max_new_tokens=max_new_tokens, device=DEV,
        context_limit=context_limit, temperature=temperature,
        repetition_penalty=repetition_penalty, no_repeat_ngram=no_repeat_ngram,
        utf8_safe=utf8_safe, seed=seed, batch_min_size=batch_min_size)
    per_row = []
    for item in outputs:
        p, r = normalize_text(item["generated"]), normalize_text(item["reference"])
        per_row.append(dict(ids=list(item["answer_ids"]), generated=item["generated"],
                            em=float(p == r), f1=char_f1(item["generated"],
                                                          item["reference"]),
                            eos=bool(item["eos"]), prompt_tokens=item["prompt_tokens"]))
    return metrics, per_row


def diagnose_split(model, tok, rows, x, starts, lengths, tag):
    """一行一套: 教师强制逐位 + 自由生成逐位 + 分歧位置。"""
    from dialogue_small import bytelevel_token_bytes, _DecodeMask
    byte_map, _ = bytelevel_token_bytes(tok)
    eos_id = tok.token_to_id("[EOS]")
    mask_cache = _DecodeMask(byte_map, eos_id, DEV)

    # ★ 规则 23: 行子集与张量包必须同源。这里取的是**前缀**, 所以强制切齐,
    #   否则 teacher_forced_rows 会按整包跑、结果与 rows 对不上(第一版就踩了)。
    n = len(rows)
    assert n <= x.shape[0], "行子集(%d)不能比张量包(%d)还大" % (n, x.shape[0])
    x, starts, lengths = x[:n], starts[:n], lengths[:n]

    tf = teacher_forced_rows(model, rows, x, starts, lengths, EVAL_BS)
    metrics, fr = free_running(model, tok, rows)

    assert len(tf) == len(fr) == len(rows), (
        "行数错位: tf=%d fr=%d rows=%d" % (len(tf), len(fr), len(rows)))

    recs = []
    for i, row in enumerate(rows):
        gold = gold_answer_ids(x, starts, lengths, i)
        pred, margin = tf[i]["pred"], tf[i]["margin"]
        assert len(pred) == len(gold) == len(margin), (
            "答案段长度错位: pred=%d gold=%d margin=%d" % (len(pred), len(gold), len(margin)))
        tf_div = next((k for k in range(len(gold)) if pred[k] != gold[k]), len(gold))
        # ★ EOS 对齐(第一版漏了, 伪造出 21 行"不闭合、margin 4.78"的假分歧):
        #   gold 末尾含 [EOS], 而 `answer_ids` **不含** EOS ⇒ 直接比 LCP 时,
        #   任何"内容全对"的行都会恰好差 1, 于是 lcp = gold_len-1 < tf_div = gold_len。
        #   吐了 EOS 就把 EOS 补回去再比。
        gen_ids = list(fr[i]["ids"]) + ([eos_id] if fr[i]["eos"] else [])
        lcp = lcp_len(gen_ids, gold)
        blocked, n_allowed = utf8_block_at_first_div(tok, mask_cache, byte_map, eos_id,
                                                     gold, tf_div)
        recs.append(dict(tag=tag, gold_len=len(gold), gen_len=len(fr[i]["ids"]),
                         tf_div=tf_div, tf_em=float(tf_div == len(gold)),
                         tf_hit=float(sum(1 for k in range(len(gold)) if pred[k] == gold[k]))
                         / max(1, len(gold)),
                         tf_margin_at_div=(margin[tf_div] if tf_div < len(margin)
                                           else float("inf")),
                         # ★分歧位(自由生成实际偏掉的那一位)的 top1-top2 logit 差。
                         #   若 L1 在这些行上不闭合, 这个数会告诉我们是"势均力敌的
                         #   数值并列被 argmax 翻盘"还是"真走了另一条路"。
                         tf_margin_at_lcp=(margin[lcp] if lcp < len(margin)
                                           else float("inf")),
                         lcp=lcp, gen_em=fr[i]["em"], gen_f1=fr[i]["f1"], eos=fr[i]["eos"],
                         utf8_blocked_at_div=bool(blocked),
                         utf8_allowed_at_div=n_allowed,
                         prompt_tokens=fr[i]["prompt_tokens"],
                         generated=fr[i]["generated"], reference=row["answer"],
                         question=row["question"]))
    return metrics, recs


# --------------------------------------------------------------------- part 0

def part0(ck, tok, tr_rows, ev_rows, tr_pack, ev_pack):
    """零训练: v11 出厂权重上的路径一致性诊断 + 两个守卫 + 一个负向对照。"""
    from model import RCfg, EngramRuleLM
    from dialogue_small import greedy_generate

    tr_x, tr_s, tr_l = tr_pack
    ev_x, ev_s, ev_l = ev_pack
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    t0 = time.time()
    m_tr, rec_tr = diagnose_split(model, tok, tr_rows[:N_ROWS], tr_x, tr_s, tr_l, "train")
    m_ev, rec_ev = diagnose_split(model, tok, ev_rows[:N_ROWS], ev_x, ev_s, ev_l, "eval")
    t_diag = time.time() - t0

    # ── G0 口径守卫: 批路径 vs 逐行路径, answer_ids 必须逐 token 相同 ─────────
    sub = tr_rows[:16]
    _, batch_rows = free_running(model, tok, sub, batch_min_size=2)
    g0 = []
    for i, row in enumerate(sub):
        g = greedy_generate(model, tok, row, max_new_tokens=MAX_NEW, device=DEV,
                            context_limit=CONTEXT_LIMIT, temperature=0.0)
        g0.append(g["answer_ids"] == batch_rows[i]["ids"])

    # ── N1 负向对照: 只改解码约束, 分歧位置**必须左移**(否则 FPD 没在测东西) ──
    _, rp_rows = free_running(model, tok, tr_rows[:N_ROWS], repetition_penalty=1.2)
    lcp_base = [r["lcp"] for r in rec_tr]
    lcp_pen = [lcp_len(rp_rows[i]["ids"],
                       gold_answer_ids(tr_x, tr_s, tr_l, i)) for i in range(N_ROWS)]

    # ── N2 负向对照: 换权重(加噪声)必须让教师强制命中率掉下来 ────────────────
    torch.manual_seed(0)
    with torch.no_grad():
        model.head.weight.add_(torch.randn_like(model.head.weight) * 0.5)
    _, rec_noisy = diagnose_split(model, tok, tr_rows[:64], tr_x, tr_s, tr_l, "train_noisy")

    return dict(
        gate=GATE, part=0, weight="v11_factory", d=int(ck["cfg"]["d"]), seed=None,
        n_rows=N_ROWS, max_new=MAX_NEW, context_limit=CONTEXT_LIMIT,
        gen_metrics_train=m_tr, gen_metrics_eval=m_ev,
        train=rec_tr, eval=rec_ev,
        g0_batch_vs_single=[bool(v) for v in g0],
        n1_lcp_base=lcp_base, n1_lcp_penalized=lcp_pen,
        n2_tf_em_noisy=[r["tf_em"] for r in rec_noisy],
        n2_tf_em_clean=[r["tf_em"] for r in rec_tr[:64]],
        diag_sec=round(t_diag, 1))


# --------------------------------------------------------------------- part 1

D_MAIN = (64, 128, 256)
SEEDS = (11, 23, 37)
STEPS = 200
LR = 0.003
KEEP_SAMPLES = 10 ** 9        # ★桃 2026-10-10「要完整问答, 不要抽样」⇒ 全留

# part1 的 train 诊断覆盖**全部 820 行**(桃的要求)。
# part0 仍是 204 行: 那是路径一致性的回归基线(已复现三次), 跑全集要多花 3 倍
# 时间换不到新信息 —— 两者的用途不同, 口径在落盘记录里分开标注。
N_ROWS_P1 = 10 ** 9


def _slim(recs):
    return [dict(r, generated=r["generated"] if i < KEEP_SAMPLES else "",
                 reference=r["reference"] if i < KEEP_SAMPLES else "",
                 question=r["question"] if i < KEEP_SAMPLES else "")
            for i, r in enumerate(recs)]


def cfg_dict(ck, d):
    """与 §48 的 build_cfg 同一口径: d 变、n_heads 跟着变, head_dim 恒 16。"""
    c = dict(ck["cfg"])
    c["d"] = d
    c["n_heads"] = d // HEAD_DIM
    return c


SAVE_CKPT = True             # d>64 要出货就必须有 checkpoint(§58.4 方案 1)


def part1(ck, tok, tr_rows, ev_rows, tr_pack, ev_pack, ds=D_MAIN, seeds=SEEDS):
    """**判决格**: 容量治好 train 侧教师强制之后, 自由生成跟不跟着好?

    训练口径与 §48 **逐字相同**(200 步 / bs=256 / lr=0.003 / seq=512 / head_dim=16),
    唯一新增的是自由生成诊断。§48 的 checkpoint 没存盘(grep 不到 torch.save),
    所以这一格只能重训 —— 这是本节最贵的一步, 也是整条链上唯一没测过的格子。

    ★**生成器**: 每跑完一格就 yield, 由 main() 立刻落盘。第一版是"跑完 9 格才
      return 整个列表", 结果中途被取消 ⇒ 25 分钟算力全丢。
    """
    import exp_capacity_scaling as ecs
    from dialogue_small import train_dialogue

    tr_x, tr_s, tr_l = tr_pack
    ev_x, ev_s, ev_l = ev_pack
    for d in ds:
        for seed in seeds:
            # ★必须走 `find_micro_batch`, 不能直接用 `start_micro_batch` 的硬编码值。
            #   理由(第��版踩了): 那个函数同时抓**两种**失败 —— 抛 OutOfMemoryError,
            #   以及"1 步超过 8 秒"这种**不抛异常**的换页抖动(§40 事故的真正症状:
            #   d=256/mb=64 不报错, 只是显存打满后 WDDM 开始换页, 进程卡住不返回)。
            #   我为了省"重建模型白跑一步"而跳过它 ⇒ 直接踩进那堵墙。
            #   实测依据(trim_padding 已生效, probe_d256_mem.py):
            #     d=256/mb=64 → 单步 7.69 s, 峰值 5922 MiB   ← 顶在 8 s 闸门边缘
            #     d=256/mb=32 → 单步 1.39~5.28 s(两次测差 3.8x, 机器被抢占), 峰值 3006 MiB
            mb, probe_peak, probe_sec = ecs.find_micro_batch(ck, d, seed, tr_pack)
            print("  d=%d seed=%d micro_batch=%d (探测单步 %.2fs / 峰值 %.0f MiB)"
                  % (d, seed, mb, probe_sec, probe_peak), flush=True)
            model = ecs.make_model(ck, d, seed, tr_x, tr_l)
            t0 = time.time()
            train_dialogue(model, tr_x, tr_s, tr_l, STEPS, ecs.BS, LR, DEV,
                           seed=seed, first_token_weight=1.0, micro_batch=mb)
            wall = time.time() - t0
            # ★训练段结束后先把缓存还给驱动, 再做诊断: 批生成的中间张量在 d=256
            #   比训练段还大(实测 1948 vs 3006 MiB), 不清就会在训练峰值上叠上去。
            if DEV == "cuda":
                torch.cuda.empty_cache()
            # ★只测 train split。桃 2026-10-10: 「以后不用跑eval」。
            #   依据不是"懒得跑", 而是 eval 侧已经被判死为**泛化问题**:
            #   §56.6 实测 d=64→128 时 eval gen_em 恒为 0/204, 容量对它零收益;
            #   §58 的结论是 eval 只能靠改数据救, 加容量是死路。
            #   ⇒ 继续测 eval 只是在反复确认一个已知不会动的数。
            m_tr, rec_tr = diagnose_split(model, tok, tr_rows[:N_ROWS_P1],
                                          tr_x, tr_s, tr_l, "train")
            n = len(rec_tr)
            # ★存盘: §48 与本节之前**所有** d>64 的 run 都只活在内存里, 跑完就没了
            #   (`exp_capacity_scaling.py` 与本脚本都没有 torch.save)。
            #   d=128 要出货就必须有 checkpoint, 否则每次问"d=128 表现如何"
            #   都得重训 97 秒。存盘后下游脚本可以直接读权重, 零训练拿读数。
            ck_path = None
            if SAVE_CKPT:
                ck_path = os.path.join(
                    HERE, "engram_rule_dialogue_bpe_answer_disjoint_d%d_s%d.pt"
                    % (d, seed))
                torch.save(dict(cfg=cfg_dict(ck, d), state_dict=model.state_dict(),
                                tokenizer=ck["tokenizer"],
                                extra=dict(raw_dialogue_rows=ck["extra"]["raw_dialogue_rows"],
                                           train_answer_starts=tr_s.tolist(),
                                           train_lengths=tr_l.tolist(),
                                           steps=STEPS, lr=LR, micro_batch=mb,
                                           seed=seed, gate=GATE)),
                           ck_path)
            print("d=%-3d seed=%-2d  %5.0fs  train: tf_em=%.4f gen_em=%.4f f1=%.4f "
                  "lcp0=%.3f%s"
                  % (d, seed, wall,
                     sum(r["tf_em"] for r in rec_tr) / n,
                     sum(r["gen_em"] for r in rec_tr) / n,
                     sum(r["gen_f1"] for r in rec_tr) / n,
                     sum(1 for r in rec_tr if r["lcp"] == 0) / n,
                     "  ckpt=%s" % os.path.basename(ck_path) if ck_path else ""),
                  flush=True)
            yield dict(gate=GATE, part=1, d=d, seed=seed, steps=STEPS, lr=LR,
                       micro_batch=mb, probe_peak_mib=round(probe_peak, 1),
                       probe_step_sec=round(probe_sec, 2),
                       n_train_rows=len(rec_tr),
                       wall_sec=round(wall, 1),
                       sec_per_step=round(wall / STEPS, 3),
                       # ★墙钟闸(只记录, 不在这里杀进程 —— 训练循环不好中断):
                       #   同配置两次实测能差 3.8 倍(机器被抢占), 单步秒数必须
                       #   带着"参考基准负载门禁"读, 见 DESIGN §43/§22。
                       slow_run=bool(wall / STEPS > ecs.PROBE_STEP_TIMEOUT_S),
                       ckpt=ck_path,
                       tf_metrics_train=m_tr,
                       train=_slim(rec_tr))
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()


# --------------------------------------------------------------------- 驱动

def append_out(rec):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def has_part(n, gate=None):
    """断点续跑: 只认**当前门禁指纹**下的记录 —— 换了代码就别拿旧数据接着跑。"""
    if not os.path.exists(OUT):
        return False
    with open(OUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if j.get("part") != n:
                continue
            if gate is None or j.get("gate") == gate:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True, choices=(0, 1))
    ap.add_argument("--ds", default=None, help="逗号分隔的 d, 默认 %s" % (D_MAIN,))
    ap.add_argument("--seeds", default=None, help="逗号分隔的 seed, 默认 %s" % (SEEDS,))
    ap.add_argument("--force", action="store_true", help="忽略断点续跑, 重新跑该 part")
    a = ap.parse_args()
    print("门禁指纹 gate = %s (exp_free_generation.py + check_free_generation_probe.py)"
          % GATE)
    # ★规则 19: 大作业前先看门狗 —— 有残留进程占着显存时直接拒跑。
    #   本轮就是在这上面栽的: 没看空闲显存就连开两档重训, 撞进 §40 的换页墙。
    if a.part == 1 and DEV == "cuda":
        free, total = torch.cuda.mem_get_info()
        free_mib, total_mib = free / (1 << 20), total / (1 << 20)
        print("启动前 GPU 空闲 %.0f / %.0f MiB (已用 %.0f)"
              % (free_mib, total_mib, total_mib - free_mib))
        if free_mib < 2048:
            raise SystemExit("空闲显存只有 %.0f MiB (<2 GiB), 先清掉残留进程再跑"
                             % free_mib)
        # ★分配器设置是**硬门禁**, 不是建议: 没开就退回了碎片化那条路(§57 实测
        #   默认分配器在 2 步内就涨过 3.5 GiB, 曾把本机 8188 MiB 的卡占满)。
        print("PYTORCH_CUDA_ALLOC_CONF = %s" % os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
        if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            raise SystemExit("必须开 expandable_segments, 否则 reserved 会碎片化涨爆显存")
    ck, tok, tr_rows, ev_rows, tr_pack, ev_pack = load_data()

    if a.part == 1:
        ds = tuple(int(x) for x in a.ds.split(",")) if a.ds else D_MAIN
        seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else SEEDS
        if has_part(1, GATE) and not a.force:
            print("当前 gate 下已有 part=1, 跳过(要重跑加 --force)")
            return 0
        # ★每跑完一格立刻落盘(part1 是生成器), 中途被取消不丢已跑的算力
        for run in part1(ck, tok, tr_rows, ev_rows, tr_pack, ev_pack, ds, seeds):
            append_out(run)
        return 0

    rec = part0(ck, tok, tr_rows, ev_rows, tr_pack, ev_pack)
    append_out(rec)
    print("part0 v11 factory: %.1fs" % rec["diag_sec"])
    for name, rs in (("train", rec["train"]), ("eval", rec["eval"])):
        n = len(rs)
        print("%-5s n=%d  tf_em=%.4f  tf_hit=%.4f  gen_em=%.4f  gen_f1=%.4f  "
              "lcp_mean=%.2f  lcp==gold_len:%.4f  lcp==0:%.4f"
              % (name, n,
                 sum(r["tf_em"] for r in rs) / n,
                 sum(r["tf_hit"] for r in rs) / n,
                 sum(r["gen_em"] for r in rs) / n,
                 sum(r["gen_f1"] for r in rs) / n,
                 sum(r["lcp"] for r in rs) / n,
                 sum(1 for r in rs if r["lcp"] == r["gold_len"]) / n,
                 sum(1 for r in rs if r["lcp"] == 0) / n))
        print("       ★L1 闭合 tf_div==lcp : %d/%d (%.4f)"
              % (sum(1 for r in rs if r["tf_div"] == r["lcp"]), n,
                 sum(1 for r in rs if r["tf_div"] == r["lcp"]) / n))
        print("       utf8 挡住首分歧位: %d/%d"
              % (sum(1 for r in rs if r["utf8_blocked_at_div"]), n))
    print("G0 批==逐行: %d/%d" % (sum(rec["g0_batch_vs_single"]), len(rec["g0_batch_vs_single"])))
    print("N1 lcp 均值 %.3f -> 重复惩罚后 %.3f"
          % (sum(rec["n1_lcp_base"]) / len(rec["n1_lcp_base"]),
             sum(rec["n1_lcp_penalized"]) / len(rec["n1_lcp_penalized"])))
    print("N2 tf_em 干净=%.4f -> 加噪=%.4f"
          % (sum(rec["n2_tf_em_clean"]) / len(rec["n2_tf_em_clean"]),
             sum(rec["n2_tf_em_noisy"]) / len(rec["n2_tf_em_noisy"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())