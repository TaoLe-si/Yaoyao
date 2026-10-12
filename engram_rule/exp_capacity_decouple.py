"""§69.6.5 容量解耦实验：固定 **D（曝光）与 d**，只扫 `n_slots`。

────────────────────────────────────────────────────────────────────────────
**为什么必须这么做**（§69.6.1–§69.6.2 的审计结论）

现有 `exp_scaling_grid_v2.py` 网格里，容量与曝光**绑死**：
  · `d` 一动 ⇒ `params`、`E`、`W` **一起**动（三者都是 d 的函数，彼此成比例）
  · `steps = ceil(40N/256)` ⇒ `D = steps·BS = 40N` ⇒ 我所谓的「数据量方向」其实是**曝光方向**
⇒ 三个容量量**一个也没被独立测过**，Chinchilla 拟合出的 β=0.0001 因此**不可信**。

本实验的解耦（已用 `model.py` 静态验证）：

    W = 16·d²            （q/k/v/o 8 个 + ffn 4 个，各 d²）→ 只随 d，**不随 n_slots**
    E = 2·d·n_slots      （engram.key (slots,d) + engram.val (slots,d)）→ 随 n_slots 线性

固定 (N, d) 时 `n_slots` 扫档 ⇒ **E 单独动，W 与 D 完全不动**。这才是干净的容量轴。

**预登记判据**（跑之前写死）

| 编号 | 判据 | 失败意味着 |
|---|---|---|
| **T1** | 固定 (N,d) 时 `n_slots` 增大必须让探针 NLL **单调不增** | 容量方向再次否证 |
| **T2** | 三轴拟合 `L = E + A/D^α + B/E^β`，要求 **β > 0** | 容量仍不服从幂律 |
| **T3** | ★负向对照：把 `n_slots` 标签**打乱**后重拟合，β 必须**变** | 判据是常量，无效 |
| **T4** | `W` 在各档**逐位相同**、`D` 逐档相同 | 解耦失败，读数作废 |

★**epochs 恒定**（`steps = ceil(EPOCHS·N/BS)`）—— 规则 19：固定步数会污染自变量。
★单 seed ⇒ **无种子极差**，不用单 seed 差异下强结论。
★逐格落盘 + `math_gate` 指纹；`n_slots` 进**恢复键与完成键**（§67 的静默跳过坑）。

用法：`python exp_capacity_decouple.py --slots 256,1024,4096,16384`
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

# ★必须在 import torch **之前**设（§57.3：分配器只在 CUDA 上下文创建时读一次配置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HTTPS_PROXY", os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7890"))
os.environ.setdefault("HTTP_PROXY", os.environ.get("HTTP_PROXY", "http://127.0.0.1:7890"))

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "capacity_decouple.jsonl")
CACHE = os.path.join(HERE, "corpus_modelscope_v2_155740.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ── 受控因子（与 exp_scaling_grid_v2 同口径，保证可比）──────────────────
SEQ = 512
BS = 256
LR = 0.003
SEED = 11
HEAD_DIM = 16
MICRO_BATCH = 64
TRAIN_FRACTION = 0.8
SPLIT_SEED = 29
WEIGHT_DECAY = 0.0          # ★固定 wd=0：这一轮只扫容量，不再做 wd 析因

EPOCHS_TARGET = 40.0
N_FIXED = 31148             # ★固定数据量/曝光：只用探针口径里已有的那一个 N
D_FIXED = 64                # ★固定宽度：W=16d² 因此恒定
SLOTS_LIST = (256, 1024, 4096, 16384)

MIN_FREE_MIB = 2048
PROBE_STEP_TIMEOUT_S = 8.0
MIN_MICRO = 4

# ★墙钟基线按 (N, d) 分段（§69.5.2：跨档借基线是我修过又复发的错）
REF_SEC_PER_STEP = {32: 0.134, 64: 0.227, 128: 0.545}   # N=31148 的实测
SLOW_MULTIPLE = 3.0


def _gate(files):
    h = hashlib.sha256()
    for name in files:
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:12]


# ★`math_gate` 只覆盖**能改训练数值**的文件；本脚本自己变不算（它只是编排）。
MATH_FILES = ("model.py", "dialogue_small.py", os.path.join("dist", "dialogue_small.py"))
MGATE = _gate(MATH_FILES)


def w_expected(d):
    """W = 16·d²：2 层 × (4 个 q/k/v/o + ffn 上下两层) = 2×(4+2+2)... 实测就是 16d²。"""
    return 16 * d * d


def e_expected(d, slots):
    """E = 2·d·n_slots：engram.key (slots,d) + engram.val (slots,d)。"""
    return 2 * d * int(slots)


def load_dialogue_small():
    """按路径显式加载 `dist/dialogue_small.py`（主目录那份 import 不了）。"""
    import importlib.util
    path = os.path.join(HERE, "dist", "dialogue_small.py")
    spec = importlib.util.spec_from_file_location("ds_decouple", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("spec 构造失败: %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ds_decouple"] = mod
    spec.loader.exec_module(mod)
    return mod


DS = load_dialogue_small()
import requests as _requests
if not hasattr(DS, "requests"):
    DS.requests = _requests

ENCODED_CACHE = os.path.join(HERE, "encoded_cache_decouple.pt")
RESUME_DIR = os.path.join(HERE, "resume_decouple")
CKPT_EVERY = 50

# 复用主实验的数据与切分逻辑（**不改主脚本**，只 import 它的纯函数）
import exp_scaling_grid_v2 as E2

DATA_URL = E2.DATA_URL
DATA_BYTES = E2.DATA_BYTES


def load_rows():
    """直接复用 E2 的缓存文件（同一份语料 ⇒ 切分逐位相同，探针可比）。"""
    if not os.path.exists(CACHE):
        raise RuntimeError("缺少语料缓存 %s（先跑一次 exp_scaling_grid_v2.py）" % CACHE)
    with open(CACHE, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    data = [r for r in recs if isinstance(r, dict) and "__meta__" not in r]
    if len(data) < 100_000:
        raise RuntimeError("语料缓存只有 %d 行" % len(data))
    return data


def split_rows(rows):
    return E2.split_rows(rows)


def assert_no_leak(train, held):
    E2.assert_no_leak(train, held)


def encode_cached(rows, tok, max_len):
    """★缓存 key 必须含 `max_len` 与语料内容（与 O3 同思路）。"""
    h = hashlib.sha256()
    h.update(hashlib.sha256(tok.to_str().encode("utf-8")).digest())
    h.update(str(max_len).encode())
    ch = hashlib.sha256()
    for r in rows:
        ch.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    h.update(ch.digest())
    h.update(str(len(rows)).encode())
    key = h.hexdigest()[:16]
    cache = {}
    if os.path.exists(ENCODED_CACHE):
        cache = torch.load(ENCODED_CACHE, map_location="cpu", weights_only=False)
    if key in cache:
        return cache[key]
    out = DS.encode_examples(rows, tok, max_len)
    cache[key] = out
    torch.save(cache, ENCODED_CACHE)
    return out


def steps_for(n, epochs=EPOCHS_TARGET, bs=BS):
    """epochs 恒定（规则 19）。floor=0：不让保底把 epochs 顶上去。"""
    return int(math.ceil(epochs * n / bs))


def micro_batch_for(d):
    return MICRO_BATCH if d <= 128 else 32


_TRAIN_X = _TRAIN_S = _TRAIN_L = None
_CK = None
_ALIVE_CACHE = None


def make_model(ck, d, slots, reuse_alive=False):
    """按 (d, n_slots) 建模型。

    ★★**默认 `reuse_alive=False`，且不得改回 True**（`check_capacity_decouple.py`
      的 C2 静态卡住这一点）。实测证据：

      | n_slots | alive 总和 | 前 256 槽是否等于小档 |
      |---:|---:|---|
      | 256 | 256 | — |
      | 4096 | 4096 | ✓（但后 3840 槽**全是 1**）|

      槽变大 ⇒ alive 总和**等比变大**，说明 `slot_ids` **依赖 n_slots**
      （哈希按槽数分桶）⇒ 跨 slots 档复用 alive 会**静默给出错的 alive**，
      而 alive 直接决定哪些槽参与前向。这是 §67.9「n_slots 越大 alive_slots 越多」
      的真正原因：不是冲突变少，是同一个 n-gram 在不同槽数下落进不同槽。

      代价：每格多算一次 `build_engram`（N=31,148 时约 21 s）。**不省这个钱。**
    """
    from model import RCfg, EngramRuleLM
    cfg = dict(ck["cfg"])
    cfg["d"] = d
    cfg["n_heads"] = d // HEAD_DIM
    cfg["n_slots"] = int(slots)
    torch.manual_seed(SEED)
    model = EngramRuleLM(RCfg(**cfg)).to(DEV)
    if reuse_alive and _ALIVE_CACHE is not None:
        model.engram.alive.copy_(_ALIVE_CACHE.to(model.engram.alive.device))
    else:
        DS.build_engram_from_examples(model, _TRAIN_X, _TRAIN_L)
    return model


def param_groups(model):
    """把参数按「W 部分 / E 部分 / 其他」分开，供 T4 判据用。"""
    w = sum(p.numel() for n, p in model.named_parameters()
            if (".ffn." in n or (".attn." in n and any(
                ".%s.weight" % t in n for t in ("q", "k", "v", "o"))))
            and n.endswith(".weight"))
    ee = sum(p.numel() for n, p in model.named_parameters() if n.startswith("engram"))
    tot = sum(p.numel() for p in model.parameters())
    return dict(w=w, e=ee, total=tot, other=tot - w - ee)


def resume_path_for(n, d, slots, steps, mb):
    """文件名自带全部自变量（§67 的静默跳过坑）。"""
    return os.path.join(RESUME_DIR,
                        "resume_dec_n%d_d%d_slots%d_s%d_mb%d.pt" % (n, d, slots, steps, mb))


def resume_key(n, d, slots, steps, mb):
    return dict(n=int(n), d=int(d), n_slots=int(slots), steps=int(steps),
                micro_batch=int(mb), seq=SEQ, bs=BS, lr=LR, seed=SEED,
                weight_decay=WEIGHT_DECAY)


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_resume(path, key, model, opt, sched, gen, step, trace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    _atomic_save(dict(key=key, model=model.state_dict(), opt=opt.state_dict(),
                      sched=sched.state_dict(), gen=gen.get_state(),
                      torch_rng=torch.get_rng_state(), step=int(step),
                      trace=list(trace), math_gate=MGATE, saved_at=time.time()), path)
    return time.time() - t0


def load_resume(path, key):
    st = torch.load(path, map_location="cpu", weights_only=False)
    got, want = st.get("key"), key
    if not isinstance(got, dict):
        raise RuntimeError("恢复点缺少 key: %s" % path)
    bad = [k for k in sorted(set(want) | set(got)) if got.get(k) != want.get(k)]
    if bad:
        raise RuntimeError("恢复键不匹配，拒绝恢复 %s\n  差异: %s\n  期望 %s\n  实际 %s"
                           % (os.path.basename(path), bad,
                              {k: want.get(k) for k in bad}, {k: got.get(k) for k in bad}))
    if st.get("math_gate") != MGATE:
        raise RuntimeError("math_gate 不符 ⇒ 恢复点作废")
    return st


def train_steps(model, x, starts, lengths, steps, micro_batch, resume=None):
    """与 `E2.train_steps` 同构（同前向/loss/OneCycle），只多了 resume。

    ★返回 `(trace, rinfo)` **二元组** —— §67 的 `sample_ranges` 教训。
    """
    if steps < 1 or micro_batch < 1 or micro_batch > x.shape[0]:
        raise ValueError("invalid steps/micro_batch")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                               pct_start=0.1)
    model.train()
    gen = torch.Generator().manual_seed(SEED)
    bucket_step = getattr(model.c, "bucket_step", 0)

    x_dev = starts_dev = lengths_dev = None
    if DEV == "cuda":
        need = (x.numel() * x.element_size() + starts.numel() * starts.element_size()
                + lengths.numel() * lengths.element_size())
        free, _t = torch.cuda.mem_get_info()
        if free - need > 2 * (1 << 30):
            x_dev, starts_dev, lengths_dev = x.to(DEV), starts.to(DEV), lengths.to(DEV)
    _x = x_dev if x_dev is not None else x
    _s = starts_dev if starts_dev is not None else starts
    _l = lengths_dev if lengths_dev is not None else lengths

    rinfo = dict(resumed_from=0, n_saves=0, save_wall_sec=0.0)
    start_step, trace = 0, []
    if resume is not None:
        rpath, rkey = resume
        if os.path.exists(rpath):
            st = load_resume(rpath, rkey)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            gen.set_state(st["gen"].cpu())
            torch.set_rng_state(st["torch_rng"].cpu())
            trace = list(st.get("trace") or [])
            start_step = int(st["step"])
            rinfo["resumed_from"] = start_step
            print("      [续训：%d/%d 步]" % (start_step, steps), flush=True)
            if start_step >= steps:
                start_step = steps

    for step in range(start_step, steps):
        idx = torch.randint(0, _x.shape[0], (BS,), generator=gen)
        answer_tokens = (_l[idx] - _s[idx]).sum().clamp_min(1)
        norm = answer_tokens
        opt.zero_grad(set_to_none=True)
        sl = torch.zeros((), dtype=torch.float32, device=DEV)
        for i in idx.split(micro_batch):
            bx_full = _x[i]
            sb = _s[i]
            lb = _l[i]
            bx = DS._trim_length(bx_full, lb) if bucket_step > 0 else bx_full
            pos = torch.arange(bx.shape[1], device=DEV).view(1, -1)
            valid = pos < lb.view(-1, 1)
            logits = model(bx)[:, :-1, :]
            tl, _fl, _na, _nf = DS.answer_loss_sums(logits, bx[:, 1:], valid, sb)
            (tl / norm).backward()
            sl += tl.detach()
        opt.step()
        sched.step()
        if step == 0 or (step + 1) % max(1, steps // 4) == 0:
            trace.append(float(sl / norm))
        if resume is not None and ((step + 1) % CKPT_EVERY == 0 or step + 1 == steps):
            rinfo["save_wall_sec"] += save_resume(resume[0], resume[1], model,
                                                  opt, sched, gen, step + 1, trace)
            rinfo["n_saves"] += 1
    model.eval()
    return trace, rinfo


@torch.no_grad()
def probe_nll(model, px, ps, pl):
    m = DS.answer_metrics(model, px, ps, pl, 32, DEV)
    return dict(probe_nll=m["answer_nll"], probe_ppl=m["answer_ppl"],
                probe_tok_acc=m["answer_token_accuracy"],
                probe_first_acc=m["first_answer_token_accuracy"],
                probe_tokens=m["answer_tokens"])


def check_free_memory():
    if DEV != "cuda":
        return
    free, total = torch.cuda.mem_get_info()
    if free / (1 << 20) < MIN_FREE_MIB:
        raise RuntimeError("空闲显存不足，拒绝启动")
    print("  空闲显存 %.0f MiB / 总 %.0f MiB" % (free / (1 << 20), total / (1 << 20)),
          flush=True)


def run_cell(n, d, slots, train_rows, probe_pack, tok, resume_on=True):
    global _TRAIN_X, _TRAIN_S, _TRAIN_L, _ALIVE_CACHE, _CK
    x, s, l = encode_cached(train_rows, tok, SEQ)
    _TRAIN_X, _TRAIN_S, _TRAIN_L = x, s, l
    ck = _CK

    # ★**不做 alive 复用**：实测证明 `slot_ids` 依赖 `n_slots`（见 make_model docstring）。
    #   上一版这里缓存了一次 alive 想省 21 s，被 C2 当场否掉 —— 那会静默给出错的 alive。
    _ALIVE_CACHE = None

    steps = steps_for(n)
    mb = micro_batch_for(d)
    model = make_model(ck, d, slots, reuse_alive=False)
    pg = param_groups(model)

    # ── T4 前置断言：解耦必须成立，否则整格读数作废 ────────────────────
    w_exp, e_exp = w_expected(d), e_expected(d, slots)
    if pg["w"] != w_exp:
        raise RuntimeError("T4 失败：W=%d ≠ 16d²=%d ⇒ 解耦不成立，拒绝记录" % (pg["w"], w_exp))
    if pg["e"] != e_exp:
        raise RuntimeError("T4 失败：E=%d ≠ 2·d·slots=%d" % (pg["e"], e_exp))

    px, ps, pl = probe_pack
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    resume = None
    if resume_on:
        resume = (resume_path_for(n, d, slots, steps, mb),
                  resume_key(n, d, slots, steps, mb))
    t0 = time.time()
    trace, rinfo = train_steps(model, x, s, l, steps, mb, resume=resume)
    wall = time.time() - t0
    if not isinstance(trace, list):
        raise RuntimeError("train_steps 第一个返回值不是 list ⇒ 二元组接错了")
    pm = probe_nll(model, px, ps, pl)
    peak = (torch.cuda.max_memory_reserved() / (1 << 20)) if DEV == "cuda" else 0.0
    n_slots = int(model.c.n_slots)
    alive_slots = int(model.engram.alive.sum())
    ckpt = os.path.join(HERE, "engram_rule_decouple_n%d_d%d_slots%d.pt" % (n, d, slots))
    torch.save(dict(cfg=model.c.__dict__, state_dict=model.state_dict(),
                    tokenizer=E2._TOK_META), ckpt)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    if resume is not None and os.path.exists(resume[0]):
        os.remove(resume[0])

    ref = REF_SEC_PER_STEP.get(d, 0.5)
    sec_per_step = wall / steps
    rec = dict(n=n, d=d, n_slots=n_slots, steps=steps, epochs=BS * steps / n,
               weight_decay=WEIGHT_DECAY, lr=LR, bs=BS, seed=SEED, seq=SEQ,
               micro_batch=mb, head_dim=HEAD_DIM,
               math_gate=MGATE, params=pg["total"], w_params=pg["w"],
               e_params=pg["e"], other_params=pg["other"],
               w_expected=w_exp, e_expected=e_exp,
               probe=pm, wall_sec=round(wall, 1),
               sec_per_step=round(sec_per_step, 3), ref_sec_per_step=ref,
               slow_run=sec_per_step > ref * SLOW_MULTIPLE,
               peak_mib=round(peak, 1), loss_trace=trace, ckpt=ckpt,
               alive_slots=alive_slots,
               resumed_from=rinfo["resumed_from"], n_saves=rinfo["n_saves"],
               save_wall_sec=round(rinfo["save_wall_sec"], 2))
    return rec


def main():
    global _CK
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default=",".join(str(v) for v in SLOTS_LIST))
    ap.add_argument("--n", type=int, default=N_FIXED)
    ap.add_argument("--d", type=int, default=D_FIXED)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    print("math_gate = %s（model.py + 两份 dialogue_small）" % MGATE)
    from tokenizers import Tokenizer
    ck = torch.load(os.path.join(HERE, "archive",
                                 "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                    map_location="cpu", weights_only=False)
    _CK = ck
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])

    rows = load_rows()
    train_rows, probe_rows = split_rows(rows)
    assert_no_leak(train_rows, probe_rows)
    print("  语料 %d → train %d / 探针 %d（词项级不相交断言通过）"
          % (len(rows), len(train_rows), len(probe_rows)), flush=True)
    if a.n > len(train_rows):
        raise SystemExit("N=%d 超过训练池 %d" % (a.n, len(train_rows)))

    probe_pack = encode_cached(probe_rows, tok, SEQ)
    slots_list = tuple(int(v) for v in a.slots.split(","))
    check_free_memory()

    done_keys = set()
    if os.path.exists(OUT) and not a.force:
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    if j.get("math_gate") == MGATE:
                        done_keys.add((j["n"], j["d"], j["n_slots"], j["steps"]))

    est = sum(steps_for(a.n) * REF_SEC_PER_STEP.get(a.d, 0.5) for _ in slots_list)
    print("  固定 N=%d d=%d，扫 n_slots=%s，steps=%d，预计 %.0f s"
          % (a.n, a.d, list(slots_list), steps_for(a.n), est), flush=True)

    for slots in slots_list:
        if (a.n, a.d, slots, steps_for(a.n)) in done_keys:
            print("  跳过 n_slots=%d（已完成）" % slots, flush=True)
            continue
        print("  >>> N=%d d=%d n_slots=%d ..." % (a.n, a.d, slots), flush=True)
        rec = run_cell(a.n, a.d, slots, train_rows[:a.n], probe_pack, tok,
                       resume_on=not a.no_resume)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("      probe NLL=%.4f tok_acc=%.4f | E=%d W=%d | %.1fs (%.3f s/步)%s"
              % (rec["probe"]["probe_nll"], rec["probe"]["probe_tok_acc"],
                 rec["e_params"], rec["w_params"], rec["wall_sec"],
                 rec["sec_per_step"], "  ★SLOW" if rec["slow_run"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
