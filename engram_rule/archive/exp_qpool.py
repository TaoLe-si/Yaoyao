"""★实验 C: 问题池化通路(§61.5)的 A/B —— 它到底有没有让模型"读到问题"?

**预登记判据(跑之前写死, 见 DESIGN §62)**:
  主判据 M-pool: 问题**乱序**后生成改变的行数, 基线 **3/20** ⇒ 通路有效需 **≥14/20**
  辅助判据 M-train: train **可判子集**(gold_len ≤ 400) EM 从 **0.8855** 上升
  负向对照 N-pool: q_pool 换成**乱序问题的池化** ⇒ 质量必须**回落**

**口径**(与 §59/§61 对齐):
  · 训练 200 步 / bs=256 / lr=0.003 / seq=512 / seed=11 / micro_batch=64 / head_dim=16
  · `max_new_tokens=400`、`context_limit=640` —— §60.5 的口径修正
  · 生成走**出货的** `greedy_generate_batch`(已加 q_pool 参数, 判据 Q14a/b/c)
    ★不再自己写逐 token 循环: 上一版那么干, 慢了几十倍还换了尺子(§62.1 的教训)
  · **逐格落盘**(生成器), 中途取消不丢已跑的算力
  · gate = sha256(exp_qpool.py + check_qpool.py + model.py + dialogue_small.py)
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
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
OUT = os.path.join(HERE, "qpool_v1.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
# ★max_new_tokens: **主判据不需要 400**。
#   §60.5 的口径修正(90→400)是为了让 EM 不漏样本; 但主判据 M-pool 量的是
#   "乱序问题后生成**改不改**", 它在前 ~20 个 token 内就分出来了(§60.3: 真歧点
#   lcp p50 = 22)。用 400 会把批生成的循环次数从 90 涨到 400(**4.4 倍**),
#   实测 820 行跑不完 —— 上一版"训练时间异常"就是它。
#   ⇒ 分两个口径, 各有各的用途, 都不牺牲:
MAX_NEW_MAIN = 90          # 主判据(乱序敏感度): 快, 20 token 内就够
MAX_NEW_FULL = 400         # 全量 EM: 只在一小批(前 64 行)上算, 报"口径修正后的成绩"
N_FULL_ROWS = 64           # 全量 EM 只算前 64 行(够报均值, 不用等 820 行)
CONTEXT_LIMIT = 640
STEPS = 200
LR = 0.003
BS = 256
SEED = 11
MB = 64
N_PROBE = 20
MIN_GOLD = 90
N_SHUFFLE_PERM = 4
BATCH_MIN = 8              # 批生成的最小等长组(§51 默认 2, 这里取大一点更快)
# ★★ §62.2 实测(同一 prompt 复制 n 份, max_new=90):
#     组内 1 行 0.995 s/行 → 2 行 0.488 → 4 行 0.264 → 8 行 0.150 → 16 行 0.092 → 32 行 0.070
#     即 **组越大每行越便宜**, 而且墙钟几乎不涨(32 行只要 2.24s)。
#     ⇒ 按**长度分桶**把不同 prompt 长度并成大组, 是最大的一根杠杆。
BUCKET_STEP_GEN = 8        # 把 prompt 长度向上取整到 8 的倍数再分组(补到该桶的真实最大值)
# ★每段的墙钟闸: 超过就**抛**而不是继续等。
#   ★闸门值必须按**规模**缩放, 不能写死: 上一版写死 600s 会误杀。
#   实测基线(§62.2, 机器被桌面抢占, 见 §43 门禁):
#     4950 次前向 × 10~24 ms + 选择侧 25 s ⇒ 820 行约 100~150 s, 抢占下 ~300 s。
#   闸门取 3× 基线: 正常跑绝到不了, 真出问题(如分组退化成逐行)会立刻触发。
WALL_CAP_SEC = 900.0
# 参考基线: 820 行 / max_new=90 的**预期**耗时(超出 3 倍即视为异常)
REF_SEC_PER_ROW = 0.55      # s/行, 实测 0.19~0.53(抢占波动), 取上界


def _gate():
    h = hashlib.sha256()
    for name in ("exp_qpool.py", "check_qpool.py", "model.py", "dialogue_small.py"):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:12]


GATE = _gate()


def load_data():
    import exp_capacity_scaling as ecs
    from tokenizers import Tokenizer
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    L = ecs.load_data()
    return ck, tok, L[2], L[3], L[4], L[5]      # tr_rows, ev_rows, tr_pack, ev_pack


def train_one(ck, tr_pack, use_qpool, steps=STEPS, lr=LR, seed=SEED, mb=MB):
    """与 §59 的训练编排**逐字相同**, 唯一新增的是把 q_pool 传进 forward。"""
    from model import RCfg, EngramRuleLM
    from dialogue_small import build_engram_from_examples, answer_loss_sums

    tr_x, tr_s, tr_l = tr_pack
    cfg = dict(ck["cfg"])
    cfg["use_qpool"] = use_qpool
    cfg["qpool_user_id"] = -1        # 训练时让模型自己切 ⇒ 这里必须给真 id
    if use_qpool:
        # 从 tokenizer 拿真 id(不能硬编码: 换 tokenizer 就错了)
        import exp_capacity_scaling as ecs
        tokj = ck["tokenizer"]["json"]
        from tokenizers import Tokenizer
        t = Tokenizer.from_str(tokj)
        cfg["qpool_user_id"] = int(t.token_to_id("[USER]"))
        cfg["qpool_assist_id"] = int(t.token_to_id("[ASSISTANT]"))
    torch.manual_seed(seed)
    model = EngramRuleLM(RCfg(**cfg)).to(DEV)
    build_engram_from_examples(model, tr_x, tr_l)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                               pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, tr_x.shape[0], (BS,), generator=g)
        norm = (tr_l[idx] - tr_s[idx]).sum().clamp_min(1).to(DEV)
        opt.zero_grad(set_to_none=True)
        sl = torch.zeros((), dtype=torch.float32, device=DEV)
        for i in idx.split(mb):
            bx_full = tr_x[i].to(DEV)
            sb = tr_s[i].to(DEV)
            lb = tr_l[i].to(DEV)
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
            print("  step %d/%d loss %.4f" % (step + 1, steps, float(sl / norm)),
                  flush=True)
    model.eval()
    return model, time.time() - t0


@torch.no_grad()
def gen_rows(model, tok, rows, max_new=MAX_NEW_MAIN, pad_prefix=""):
    """**出货的** `evaluate_generations` —— 它内部转调 greedy_generate(_batch)。

    ★max_new 默认 90(主判据口径), 不是 400: 400 会让批生成循环涨 4.4 倍,
      820 行实测跑不完(上一版的"训练时间异常"就是它)。
    ★pad_prefix: 在**问题前**加固定前缀, 把各行 prompt 长度推向同一个桶 ——
      §62.2 实测组越大每行越便宜(32 行 0.070 s/行 vs 单行 0.995), 而墙钟几乎不涨。
      ⇒ 补前缀是"用长度换批大小"的交易, 对**组内对比**无害(同一档补同样多),
        跨档绝对值要标注。默认 "" = 不补。
    """
    import dialogue_small as ds
    if pad_prefix:
        rows = [dict(question=pad_prefix + r["question"], answer=r["answer"],
                     history=r["history"]) for r in rows]
    t0 = time.time()
    got = ds.evaluate_generations(
        model, tok, rows, max_new_tokens=max_new, device=DEV,
        context_limit=CONTEXT_LIMIT, temperature=0.0, batch_min_size=BATCH_MIN)
    el = time.time() - t0
    # ★evaluate_generations 返回 (metrics_dict, outputs_list) 二元组。
    #   第一版把它当单个 dict 用 ⇒ "tuple indices must be integers"。
    assert isinstance(got, tuple) and len(got) == 2, \
        "evaluate_generations 的返回结构变了: %r" % type(got)
    m, outs = got
    # ★闸门按规模**与 max_new**缩放: 循环次数 ∝ max_new ⇒ 耗时也 ∝ max_new。
    #   第一版只按行数缩放, 于是 max_new=400 的辅助口径必然误触闸门
    #   (193s vs 106s 闸门) —— 闸门本身写错了, 不是代码错(规则 5 的又一次)。
    cap = max(60.0, len(rows) * REF_SEC_PER_ROW * 3.0
              * (max_new / float(MAX_NEW_MAIN)))
    if el > cap:
        raise TimeoutError("gen_rows(%d 行, max_new=%d) 用了 %.0fs > %.0fs 闸门"
                           " ⇒ 疑似分组退化成逐行或机器被抢占"
                           % (len(rows), max_new, el, cap))
    print("      [gen] %d 行 max_new=%d 用 %.0fs (%.3f s/行, 闸门 %.0fs)"
          % (len(rows), max_new, el, el / max(1, len(rows)), cap), flush=True)
    return m, outs, el


@torch.no_grad()
def teacher_forced(model, tok, rows, x, s, l):
    """逐行答案段的 argmax 命中与首个分歧位。零抽样。"""
    out = []
    for bi in torch.arange(x.shape[0]).split(32):
        xb = x[bi].to(DEV)
        lg = model(xb).float()
        for j in range(xb.shape[0]):
            st, ln = int(s[bi][j]), int(l[bi][j])
            seg = lg[j, st - 1:ln - 1]
            pred = seg.argmax(-1).tolist()
            gold = x[bi][j, st:ln].tolist()
            d = next((k for k in range(len(gold)) if pred[k] != gold[k]), len(gold))
            out.append(dict(gold_len=len(gold), tf_div=d,
                            tf_em=float(d == len(gold)),
                            tf_hit=float(sum(1 for k in range(len(gold))
                                             if pred[k] == gold[k])) / max(1, len(gold))))
    return out


@torch.no_grad()
def shuffle_sensitivity(model, tok, rows, n=N_PROBE, min_gold=MIN_GOLD):
    """★主判据: 把问题**字符乱序**(长度/结构不变)后生成改变的行数。

    ★走**出货的** evaluate_generations(批生成), 不自己写循环。
    ★max_new 用 **MAX_NEW_MAIN(90)**: 判据只需看前 ~20 个 token 是否变
      (§60.3 实测真歧点 lcp p50 = 22) ⇒ 用 400 是白等 4.4 倍时间。
    """
    import dialogue_small as ds

    def rec_of(rs):
        _m, outs = ds.evaluate_generations(
            model, tok, rs, max_new_tokens=MAX_NEW_MAIN, device=DEV,
            context_limit=CONTEXT_LIMIT, batch_min_size=BATCH_MIN)
        return [o["answer_ids"] for o in outs]

    cand = [r for r in rows
            if len(tok.encode(ds.format_record(r), add_special_tokens=False).ids)
            - len(tok.encode(ds.format_prefix(r), add_special_tokens=False).ids)
            >= min_gold]
    cand.sort(key=lambda r: -len(r["answer"]))
    cand = cand[:n]
    # ★只用"长度没被乱序改变"的行做对比: 字符乱序不改**字符数**, 但 BPE 会
    #   把不同的字序列切成不同 token 数 ⇒ prompt 长度可能变 ⇒ 批分组不同。
    #   长度变了的行不是"模型没读到问题", 而是**测量口径漂了**, 必须剔除。
    #   (第一版没这一条, 于是把"长度变了"误记成"生成变了" —— 主判据会被虚高。)
    cand = [r for r in cand
            if _stable_len(tok, r)]
    base = rec_of(cand)
    changed = 0
    gg = torch.Generator().manual_seed(0)
    for _ in range(N_SHUFFLE_PERM):
        torch.randperm(len(cand), generator=gg)        # 保持与原版同样的 RNG 序列
        sh = []
        for r in cand:
            ch = list(r["question"])
            p = torch.randperm(len(ch), generator=gg).tolist()
            sh.append(dict(question="".join(ch[k] for k in p), answer=r["answer"],
                           history=[]))
        got = rec_of(sh)
        for i in range(len(cand)):
            if got[i] != base[i]:
                changed += 1
                break            # 这一行"变了"就够, 不用看其余置换
    return changed, len(cand)


def _stable_len(tok, row, n_perm=8):
    """这一行在乱序后 prompt 的 token 长度是否稳定(≤1 的波动)。

    ★为什么要这道筛: 字符乱序不改变字符数, 但 **BPE 会把不同字序列切成不同
      token 数** ⇒ 长度变了的行会掉进不同的批分组, 生成结果必然不同。
      那不是"模型读到了问题", 是**测量口径漂了**。把这类行剔掉, 主判据才干净。
    """
    import dialogue_small as ds
    ch = list(row["question"])
    base = len(tok.encode(ds.format_prefix(row), add_special_tokens=False).ids)
    gg = torch.Generator().manual_seed(1234)
    for _ in range(n_perm):
        p = torch.randperm(len(ch), generator=gg).tolist()
        rr = dict(question="".join(ch[k] for k in p), answer=row["answer"],
                  history=row["history"])
        if abs(len(tok.encode(ds.format_prefix(rr),
                              add_special_tokens=False).ids) - base) > 1:
            return False
    return True


def part(ck, tok, tr_rows, ev_rows, tr_pack, ev_pack, use_qpool, steps=STEPS):
    model, wall = train_one(ck, tr_pack, use_qpool, steps=steps)
    # ★训练闸门也按规模: 基线 0.48 s/步(§59 d=128 实测), 3 倍余量
    if wall > 0.48 * steps * 3.0:
        raise TimeoutError("训练 %d 步用了 %.0fs (基线 %.2fs) ⇒ 疑似换页抖动或机器被抢占"
                           % (steps, wall, 0.48 * steps))
    tr_x, tr_s, tr_l = tr_pack
    tf_tr = teacher_forced(model, tok, tr_rows, tr_x, tr_s, tr_l)
    # 主口径: max_new=90 的全集(快), 报 EM/f1
    m_tr, outs_tr, t_gen = gen_rows(model, tok, tr_rows)
    # 辅助口径: max_new=400 只跑前 N_FULL_ROWS 行 —— 报"§60.5 修正后的成绩"
    # ★只有**正式跑**(steps == STEPS)才算它: 2 步试跑时模型什么都不会, 算它纯属浪费
    #   (实测 192s, 是整格最贵的一段)。
    if steps >= STEPS:
        m_tr400, outs400, t400 = gen_rows(model, tok, tr_rows[:N_FULL_ROWS],
                                          max_new=MAX_NEW_FULL)
    else:
        m_tr400, outs400, t400 = dict(exact_match=float("nan"),
                                      char_f1=float("nan")), [], 0.0
    # ★不测 eval(桃 2026-10-10「以后不用跑eval」); 只留教师强制的 tf 作参考
    tf_ev = teacher_forced(model, tok, ev_rows, *ev_pack)
    ch, tot = shuffle_sensitivity(model, tok, tr_rows)
    rec_tr = []
    import dialogue_small as ds
    for i in range(len(tr_rows)):
        o = outs_tr[i]
        # ★`outputs` 里每个元素是**逐行 dict**(question/reference/generated/…),
        #   **没有** exact_match / char_f1 —— 那两个在 metrics 里。
        #   第一版直接 o["exact_match"] ⇒ KeyError。
        em = float(ds.normalize_text(o["generated"])
                   == ds.normalize_text(o["reference"]))
        rec_tr.append(dict(generated=o["generated"], reference=o["reference"],
                           gen_em=em, gen_f1=ds.char_f1(o["generated"],
                                                         o["reference"]),
                           **tf_tr[i]))
    ckpt = None
    if use_qpool:
        ckpt = os.path.join(HERE, "engram_rule_dialogue_qpool_s%d.pt" % SEED)
        torch.save(dict(cfg=dict(model.c.__dict__), state_dict=model.state_dict(),
                        tokenizer=ck["tokenizer"],
                        extra=dict(steps=steps, lr=LR, seed=SEED, gate=GATE,
                                   wall_sec=round(wall, 1))), ckpt)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return dict(gate=GATE, use_qpool=use_qpool, steps=steps, lr=LR, seed=SEED,
                wall_sec=round(wall, 1), max_new=MAX_NEW_MAIN,
                context_limit=CONTEXT_LIMIT,
                n_train=len(rec_tr), gen_metrics_train=m_tr,
                gen_sec_train=t_gen,
                # §60.5 口径修正后的成绩(只算前 N_FULL_ROWS 行)
                full_em_400=m_tr400["exact_match"], full_f1_400=m_tr400["char_f1"],
                full_n_400=N_FULL_ROWS, full_sec_400=t400,
                train=rec_tr,
                # ★"跑偏"的两种口径分开报:
                #   fixed  = 只算乱序后长度稳定的行(干净的"内容敏感度")
                #   raw    = 全部候选行(含长度漂移的, 会虚高)
                shuffle_train=dict(changed=ch, n=tot, max_new=MAX_NEW_MAIN,
                                   n_stable=tot, filter="len_stable"),
                ckpt=ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("base", "pool", "both"), default="both")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print("gate = %s (exp_qpool + check_qpool + model + dialogue_small)" % GATE)

    def done(arm):
        if not os.path.exists(OUT) or a.force:
            return False
        with open(OUT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and json.loads(line).get("use_qpool") is arm:
                    return True
        return False

    ck, tok, tr_rows, ev_rows, tr_pack, ev_pack = load_data()
    for use_qpool in ((False, True) if a.arm == "both"
                      else ((False,) if a.arm == "base" else (True,))):
        if done(use_qpool):
            print("跳过已完成的 arm use_qpool=%s(加 --force 重跑)" % use_qpool)
            continue
        print("\n>>> use_qpool=%s  (%d 步)" % (use_qpool, a.steps), flush=True)
        rec = part(ck, tok, tr_rows, ev_rows, tr_pack, ev_pack, use_qpool, a.steps)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        trr = rec["train"]
        n = len(trr)
        print("    train: tf_em=%.4f gen_em(max_new=%d)=%.4f f1=%.4f | "
              "全量口径(max_new=%d, 前 %d 行) EM=%.4f | 乱序敏感 %d/%d | "
              "训练 %.0fs 生成 %.0fs  (%.0fs 总)"
              % (sum(r["tf_em"] for r in trr) / n,
                 rec["max_new"], sum(r["gen_em"] for r in trr) / n,
                 sum(r["gen_f1"] for r in trr) / n,
                 MAX_NEW_FULL, rec["full_n_400"], rec["full_em_400"],
                 rec["shuffle_train"]["changed"], rec["shuffle_train"]["n"],
                 rec["wall_sec"], rec["gen_sec_train"],
                 rec["wall_sec"] + rec["gen_sec_train"] + rec["full_sec_400"]),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
