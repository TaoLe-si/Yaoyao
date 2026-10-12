"""★实验 F（v2 网格）：把「数据量」与「容量」的效应**分离**后再测 scaling。

────────────────────────────────────────────────────────────────────────────
**为什么必须重做（v1 的三处混淆，全部有落盘证据）**

`scaling_grid_v1.jsonl` 把 `steps=200` 钉死，只变 N 与 d。后果是每行的重复次数
`epochs = bs·steps / N` 随 N 反比：

| N | epochs | train NLL d=32→128 | 探针 NLL 变化 |
|---:|---:|---:|---:|
| 50 | **1024.0** | 0.0990 → 0.0213 | **+0.2191** |
| 100 | 512.0 | 0.1254 → 0.0133 | **+0.2798** |
| 200 | 256.0 | 0.3078 → 0.0100 | **+0.3844** |
| 400 | 128.0 | 1.1954 → 0.0160 | **+0.3649** |
| 738 | **69.4** | 2.5228 → 0.0710 | **+0.9909** |

⇒ N 变小的同时「过拟合程度」也变，两个变量同时动。v1 报的「加容量一致让探针变差」
**无法归因到容量**。

混淆 2：`dialogue_small.py:417` 把 `AdamW(weight_decay=0.01)` 写死。d 从 32 到 128
参数量涨 4.3 倍，衰减力随之放大 —— 这会**人为压低宽模型的留出表现**。
⇒ v2 必须在**同格**做 wd=0 / wd=0.01 的 2×2 析因。

混淆 3：v1 的数据只有 1024 行（train 池 738）。桃已批准扩数据，且 §49/§58 已把
泛化侧的约束指向数据量。

**v2 的三个受控改动（只动实验脚本，不动模型、不动出货训练路径）**

1. `steps = ceil(EPOCHS_TARGET · N / bs)` ⇒ **epochs 恒定**，只留 N 一个变量。
2. `weight_decay ∈ {0.0, 0.01}` 作为显式因子进网格。
3. 数据从 ModelScope **固定 revision** 拉更大的有界样本（上限 4096 行），
   全量去重 + 词项级 train/探针不相交断言。

**预登记判据（跑之前写死，见 DESIGN §67）**

- **S1**：修好 epochs 与 wd 混淆后，固定 N 时 `d` 增大必须让探针 NLL **单调不增**。
- **S2**：固定 d 时 `N` 增大必须让探针 NLL **单调不增**。
- **S3**：`wd=0` 与 `wd=0.01` 两臂的差异必须**显著小于**容量效应本身，
  否则说明 v1 的「容量符号为负」是衰减造成的假象（这条是 v2 的核心诊断）。
- 任一必要条件不满足 ⇒ 判否证，**不硬凑**。

**口径纪律**
- bs=256（规则 18）、micro_batch=64、lr=0.003、head_dim=16、seed=11、seq=512。
- 探针集 = `split_records(train_fraction=0.8, seed=29)` 的 heldout，**任何一格都不训练它**。
- 单 seed（桃的指示）⇒ 无种子极差，不报置信区间。
- 逐格落盘 + 双指纹（§69.3）：`gate` 覆盖本脚本（编排一改就变），
  `math_gate` 只覆盖 `model.py` + 两份 `dialogue_small.py`（**只有它才能改训练数值**）。
  完成判定走 `gate_ok()`，旧 `gate` 靠**显式**等价声明过渡。
- **断点续训（§69.1）**：每 `CKPT_EVERY=50` 步原子存一次恢复点（model/opt/sched/
  `gen`/全局 RNG/步数/trace）。恢复键覆盖全部自变量，任一不符即拒绝恢复。
- 墙钟闸：单步超 `PROBE_STEP_TIMEOUT_S` 秒 ⇒ 视作换页抖动，**自动降 micro_batch 重来**
  （§57.4 的教训：显存打满不抛异常，只表现为极慢）。
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

OUT = os.path.join(HERE, "scaling_grid_v2.jsonl")
CACHE = os.path.join(HERE, "corpus_modelscope_v2_155740.jsonl")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ── 受控因子 ────────────────────────────────────────────────────────────
SEQ = 512
BS = 256
LR = 0.003
SEED = 11
HEAD_DIM = 16
MICRO_BATCH = 64
TRAIN_FRACTION = 0.8
SPLIT_SEED = 29

EPOCHS_TARGET = 40.0        # ★v2 的核心：epochs 恒定，只让 N 动
MIN_STEPS = 1               # 真正的硬下限（1 步）；epochs 地板由 EPOCHS_TARGET 决定
FLOOR_STEPS = 200           # 只为「与 v1 可比」设的软地板，可用 --no-floor 关掉
WEIGHT_DECAY_SHIPPING = 0.01   # 出货路径写死的那个值（dialogue_small.py:417）

N_LIST = (1280, 2000, 2952)
D_LIST = (32, 64, 128)
WD_LIST = (0.0, WEIGHT_DECAY_SHIPPING)

# ★墙钟基线：**不是常数**。v1 的 REF_SEC_PER_STEP 取自 N<=738 的小语料，
#   而 v2 的 N 大到 124,593 ⇒ 每步要扫更多行、算更多 token，s/步必然随 N 上升。
#   我把小语料的常数当基线，于是每格都「超基线」，但阈值 3x 从未触发 ⇒ 判据失效。
#   现改为：基线 = f(d) 的基础值 再乘一个随 N 增长的修正因子。
#   N 修正的推导：步耗时主要由「本步处理的答案 token 数」决定，而
#   steps ∝ N（epochs 恒定）且每批固定抽 BS=256 行 ⇒ 每步的 token 数大致不随 N 变
#   （批大小固定）。真正随 N 增长的是**累计工作量**，不是单步。
#   所以单步 s/步应近似与 N 无关；实测从 0.088(N=4k) 涨到 0.182(N=124k) 的 2.07x，
#   来自 N 变大后**单批内长度分布更分散**（截断长度更大 ⇒ 每步算的 padding 更多）。
#   ⇒ 基线按 N 分段给出实测值，判据改为「与同 N 档实测比」，不跨 N 比。
REF_SEC_PER_STEP_BASE = {32: 0.109, 64: 0.156, 128: 0.258}   # N<=738 的 v1 实测
# 按 N 档的实测基线（本脚本自己跑出来的，见 scaling_grid_v2.jsonl 的 sec_per_step）
REF_SEC_PER_STEP_BY_N = {
    4096:   {32: 0.088, 64: None,  128: None},
    12459:  {32: 0.129, 64: None,  128: None},
    31148:  {32: 0.134, 64: 0.227, 128: 0.545},
    124593: {32: 0.182, 64: 0.244, 128: None},
}
SLOW_MULTIPLE = 3.0     # 单步超同档基线 3x ⇒ 判慢（§57.4/§67.10）
PROBE_STEP_TIMEOUT_S = 8.0      # §57.4：单步超时 ⇒ 疑似换页抖动
MIN_MICRO = 4
MIN_FREE_MIB = 2048             # §57.5 启动闸：空闲显存不足就拒绝启动

# ── 断点续训（§69.1）──────────────────────────────────────────────────
CKPT_EVERY = 50                 # 每 50 步原子存一次恢复点（§69.1 的 P5 要实测占比）
RESUME_DIR = os.path.join(HERE, "resume_v2")

# ModelScope 固定 revision（与 v11 同一份数据集，只把有界样本放大）
DATA_URL = ("https://www.modelscope.cn/datasets/qiaojiedongfeng/qiaojiedongfeng/"
            "resolve/c0adb8dbaaeddf3b83f0e4f4cf95eb074c597610/train.jsonl")
DATA_BYTES = 58_891_185
MAX_BYTES = 16_777_216
MAX_LINES = 4096


def _gate():
    """指纹覆盖：本脚本 + model.py + 两份 dialogue_small.py。

    ★主目录那份 `dialogue_small.py` **import 不了**（它 `import train`，而 train.py
      已归档、archive 下又没有它要的 baseline 包）。实际加载的是 `dist/` 那份
      （逐字相同、只少 3 行 import，已 difflib 核实）。两份都要进指纹。
    """
    h = hashlib.sha256()
    for name in ("exp_scaling_grid_v2.py", "model.py", "dialogue_small.py",
                 os.path.join("dist", "dialogue_small.py")):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:12]


GATE = _gate()


# ★§69.3：`gate` 把**实验编排脚本本身**也算进指纹，于是「加断点续训」这种
#   不改任何训练数值的改动也会作废全部读数 —— 实测会作废 7 格，其中一格 4758 s。
#   ⇒ 拆出 `math_gate`：**只覆盖能改变训练数值的文件**。
#   ★「编排改动不改数值」**不是假设**，是靠 `check_resume.py` 的 P1 逐位对拍担保的；
#     担保不成立就整条作废（见 LEGACY_GATE_EQUIV 里的绑定）。
MATH_FILES = ("model.py", "dialogue_small.py", os.path.join("dist", "dialogue_small.py"))


def _math_gate():
    h = hashlib.sha256()
    for name in MATH_FILES:
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:12]


MGATE = _math_gate()

# ★旧 `gate` → 新 `math_gate` 的**显式**等价声明；新增一条必须同时写清担保它的证据。
# ★★2026-10-11 修正：原来这里写的是 `math_gate=MGATE`，那是**空绑定**。
#   `gate_ok` 判的是 `LEGACY_GATE_EQUIV["math_gate"] == MGATE`，而右边的 MGATE
#   正是这里赋进去的同一个值 ⇒ **恒真**，无论 model.py 改多少都不会失效。
#   注释里写的「MGATE 一旦变…等价声明自动失效」**与实现相反**，是假绿。
#   ⇒ 改成**字面量**：钉住当初做这份声明时的那套代码指纹。
LEGACY_GATE_EQUIV = dict(
    math_gate="4faabefdcd58",          # ★字面量，绝不能写 MGATE（那是恒真绑定）
    gates=("ff5eefd1604b", "59855015dfd8"),
    evidence="§69.3 的 P1：直跑 12 步 vs 6 步+存+恢复+6 步，"
             "state_dict 逐位相同、trace 逐点相同（max|Δ| = 0）",
)

# ★2026-10-11：`dist/dialogue_small.py` 给 `answer_loss_sums` 加了可选 think 段
#   ⇒ MGATE 由 4faabefdcd58 变成 eb7136d1f3f6。但**出货路径从不传 think_start**，
#   四个返回值逐位不变（判据 check_think_backcompat.py 的 B1/B2），旧读数仍然有效。
#   ★两端都钉**字面量**：`dst` 不等于当前 MGATE 时这条迁移整体失效 ⇒
#     之后再真改了训练数值，它不会跟着漂移（正是旧绑定栽的坑）。
NEUTRAL_MIGRATIONS = (
    dict(src="4faabefdcd58", dst="eb7136d1f3f6",
         why="dist/dialogue_small.py: answer_loss_sums 增加可选 think_start/think_weight；"
             "think_start=None 时与旧实现逐位相同",
         evidence="check_think_backcompat.py：B1 跨 40 组随机输入（含 64×511 训练实况形状）"
                  "四个返回值 max|Δ| = 0.000e+00；B2 负向对照 max|Δ| = 8.43e+02 "
                  "（证明该比较确实敏感，不是恒真）"),
)


def _mgate_equivalent(mg):
    """`mg` 与**当前** MGATE 在训练数值上是否等价（含已登记的中性迁移）。

    ★每一步都是**真实比较**：没有任何一处拿「当前值」和「当前值」比。
      判据的可失败性由 check_gate_binding.py 的 G2/G3 验证。
    """
    if mg == MGATE:
        return True
    return any(mv["src"] == mg and mv["dst"] == MGATE for mv in NEUTRAL_MIGRATIONS)


def gate_ok(rec):
    """这条记录是否由**当前这套训练数值代码**产生（§69.3）。

    新记录带 `math_gate` ⇒ 直接比，不等价时看有没有登记过的中性迁移。
    旧记录只有 `gate` ⇒ 必须命中 `LEGACY_GATE_EQUIV["gates"]`，并把它的 `gate`
    **翻译**成当初那套数学代码的 math_gate（字面量），再走同一条等价判定。
    ★这样旧记录也能跟着中性迁移走，否则它们会在代码变一次后被误作废。
    """
    mg = rec.get("math_gate")
    if mg is None:
        if rec.get("gate") not in LEGACY_GATE_EQUIV["gates"]:
            return False
        mg = LEGACY_GATE_EQUIV["math_gate"]      # ★字面量，不是 MGATE
    return _mgate_equivalent(mg)


def load_dialogue_small():
    """按路径显式加载 `dist/dialogue_small.py`（理由见 module docstring 的门禁说明）。"""
    import importlib.util
    path = os.path.join(HERE, "dist", "dialogue_small.py")
    spec = importlib.util.spec_from_file_location("ds_offline_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("spec 构造失败: %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ds_offline_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


DS = load_dialogue_small()

# ★`dist/dialogue_small.py` 是离线副本，**刻意删掉了 `import requests`**（只少这 3 行：
#   requests / train_module / from train import）。所以 `DS.sample_ranges` 里那个
#   `requests.Session()` 会 NameError。v2 要联网拉数据，这里把 requests 显式补回去 ——
#   **只注入模块属性，不改 dist 的源文件**（它是交付出去的离线包，一个字都不动）。
import requests as _requests
if not hasattr(DS, "requests"):
    DS.requests = _requests



ENCODED_CACHE = os.path.join(HERE, "encoded_cache_v2.pt")


def encode_cached(rows, tok, max_len):
    """★优化 3：BPE 编码结果缓存到磁盘。

    124,593 行逐行编码要 11 s，而**同一个 (语料, tokenizer, max_len) 的编码结果是确定的**。
    多格扫描时这 11 s 每格都白花。缓存 key 用「训练行内容 + tokenizer + max_len」的 sha256，
    任何一项变了就自动失效（与 gate 指纹同一思路）。
    """
    import hashlib as _hl
    tok_json = tok.to_str() if hasattr(tok, "to_str") else ""
    h = _hl.sha256()
    h.update(_hl.sha256(tok_json.encode("utf-8")).digest())
    h.update(str(max_len).encode())
    # ★key 必须含**语料内容**哈希，只靠 len(rows) 会在换语料但行数相同时误命中
    import json as _json
    _ch = _hl.sha256()
    for _r in rows:
        _ch.update(_json.dumps(_r, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    h.update(_ch.digest())
    h.update(str(len(rows)).encode())
    key = h.hexdigest()[:16]
    cache = {}
    if os.path.exists(ENCODED_CACHE):
        cache = torch.load(ENCODED_CACHE, map_location="cpu", weights_only=False)
    if key in cache:
        print("      [编码缓存命中 %s，跳过 %d 行 BPE]" % (key, len(rows)), flush=True)
        return cache[key]
    out = DS.encode_examples(rows, tok, max_len)
    cache[key] = out
    torch.save(cache, ENCODED_CACHE)
    return out

def normalize_key(text):
    """与出货同一套归一化（NFKC + casefold + 去空白），保证「词项级」比较同口径。"""
    return DS.normalize_text(text)


# ── 数据 ────────────────────────────────────────────────────────────────

def load_rows():
    """拉/读 ModelScope 的有界样本，返回 `list[dict(question,answer,history)]`。

    ★`DS.sample_ranges` 的返回是 **`(rows, info)` 二元组**，不是单个 list ——
      我第一版按 list 处理，落盘把整个 list 塞进一行 JSON，读回来只有 3 行，
      `split_records` 于是报 "needs >=4 rows"。这是**调用方不看被调方签名**的错，
      不是数据源的问题。
    """
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        data = [r for r in recs if isinstance(r, dict) and "__meta__" not in r]
        if len(data) >= 100_000:
            return data
        print("  !! 缓存只有 %d 行可用，重新拉取" % len(data), flush=True)
    rows, info = DS.sample_ranges(url=DATA_URL, total_bytes=DATA_BYTES,
                                  max_bytes=MAX_BYTES, max_lines=MAX_LINES,
                                  rows_per_range=1024)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("sample_ranges 返回异常: rows=%s" % type(rows).__name__)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    meta = dict(rows=len(rows), url=DATA_URL,
                revision=DATA_URL.rsplit("/", 1)[-1], sha256=digest.hexdigest(),
                info=info if isinstance(info, dict) else {"info_type": type(info).__name__})
    # ★一行一条 row；meta 单独一行，读的时候按 `__meta__` 过滤掉
    with open(CACHE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.write(json.dumps({"__meta__": meta}, ensure_ascii=False, sort_keys=True) + "\n")
    print("  已拉取并缓存 %d 行 → %s（sha256=%s…）"
          % (len(rows), os.path.basename(CACHE), meta["sha256"][:12]), flush=True)
    return rows


def split_rows(rows):
    """question/answer 连通分量切分：train 与 heldout（heldout 作固定探针）。"""
    data = [r for r in rows if "__meta__" not in r]
    train, held = DS.split_records(data, train_fraction=TRAIN_FRACTION,
                                   seed=SPLIT_SEED, return_meta=False)
    return train, held


def assert_no_leak(train, held):
    """★全量、词项级（规则 11）。任何交集都必须拒绝，不做截断检查。

    ★去重口径（2026-10-11 修正，此前**写错**过一次）：按**原始** `(question, answer)`
      判重，**不**按 normalize_key 判。原因：155,740 行大语料里有 33 条行在
      NFKC+casefold+去空白后与另一行撞键（全角/半角、大小写等细微差异）——
      那是数据本身的**近重复**，不是泄漏，更不是「重复行」。
      真正的泄漏判据是下面那两个**集合交集**（train 与 held 的 question/answer 集合），
      实测 `train ∩ probe = 0`。
    """
    tq = {normalize_key(r["question"]) for r in train}
    ta = {normalize_key(r["answer"]) for r in train}
    hq = {normalize_key(r["question"]) for r in held}
    ha = {normalize_key(r["answer"]) for r in held}
    if tq & hq:
        raise AssertionError("question 泄漏 %d 项: %s" % (len(tq & hq), sorted(tq & hq)[:5]))
    if ta & ha:
        raise AssertionError("answer 泄漏 %d 项: %s" % (len(ta & ha), sorted(ta & ha)[:5]))
    # 全量去重：按原始文本判（归一化撞键是近重复，不算重复行）
    keys = [(r["question"], r["answer"], tuple(r["history"]))
            for r in list(train) + list(held)]
    if len(keys) != len(set(keys)):
        dup = len(keys) - len(set(keys))
        raise AssertionError("存在 %d 条原始文本重复行" % dup)


# ── 网格 ────────────────────────────────────────────────────────────────

def steps_for(n, epochs=EPOCHS_TARGET, bs=BS, floor=FLOOR_STEPS):
    """epochs 恒定 ⇒ steps ∝ N；受整数量化影响 epochs ∈ [epochs, epochs+bs/N)。

    ★floor=0 时完全按 epochs 算，不再有保底步数把 epochs 顶上去的规则 ——
      这才让「epochs 恒定」成为可检验不变量而不是近似。
      默认 floor=200 只为与 v1 可比（v1 就是 200 步）；--no-floor 关掉。
    """
    return max(floor, int(math.ceil(epochs * n / bs)))


def build_grid(ns=N_LIST, ds=D_LIST, wds=WD_LIST, floor=FLOOR_STEPS, ep_target=EPOCHS_TARGET):
    cells = []
    for n in ns:
        steps = steps_for(n, epochs=ep_target, floor=floor)
        for d in ds:
            for wd in wds:
                cells.append(dict(n=n, d=d, steps=steps, weight_decay=wd,
                                  lr=LR, bs=BS, seed=SEED, head_dim=HEAD_DIM,
                                  micro_batch=MICRO_BATCH, seq=SEQ,
                                  epochs=BS * steps / n))
    return cells


# ── 训练 ────────────────────────────────────────────────────────────────

def ref_for(n, d):
    """取这一 (N, d) 的墙钟基线；没有实测值就退回 v1 的 d 基础值。

    ★这是修 2026-10-11 的「速度异常」：旧版用 v1 小语料的**常数**基线，
      跨 N 比较必然误判（N 从 4k 到 124k，实测 s/步 涨 2.07x）。
      现在**按 N 档查表**，同档比同档。
    """
    row = REF_SEC_PER_STEP_BY_N.get(n, {})
    v = row.get(d)
    return float(v) if v else float(REF_SEC_PER_STEP_BASE.get(d, 0.5))


def micro_batch_for(d):
    """按 d 分档给 micro_batch。

    ★2026-10-12 修（静态可预见的缺陷，不是试错出来的）：
      原实现只有两档（d<=128 -> 64，否则 32）。d=512 时 `n_heads = d//16 = 32`
      ⇒ SwiGLU 的 `gate`/`up` 中间张量 (B,L,W,H,Dg) 与 **H** 成正比，
      d=512 是 d=256 的两倍 ⇒ 张量也两倍 ⇒ mb=32 直接
      `CUDA driver error: out of memory`（model.py:482 的 `gate.abs()`）。
      ⇒ 中间张量大小 ∝ micro_batch × n_heads = micro_batch × d/16
        （AGENTS 规则 26 的 B·W 自适应同一族），所以按 **d/16 单调降**：
          d<=128  -> 64     (H=8)
          d<=256  -> 32     (H=16)
          d>256   -> 16     (H>=32)
    ★两道自动闸都没覆盖这个场景：启动时显存够（规则 28 的 2 GiB 闸不响），
      而 OOM 是**立即抛错**、不是变慢（8 s 超时闸也不响）—— 与 §57.4 那种
      「不抛 OOM 只表现为极慢」是两种形态。这处缺陷只能静态修。
    """
    if d <= 128:
        return MICRO_BATCH          # 64
    if d <= 256:
        return 32
    return 16



_ALIVE_CACHE = None      # ★已算好的 engram.alive（同训练集 ⇒ 确定），跨模型复用

def make_model(ck, d, slots=None, reuse_alive=False):
    """按 d 构造模型；n_heads 随 d 变、head_dim 恒为 16（§48 的口径）。

    ★`slots` 覆盖 n_slots：换一个「容量」维度做 A/B（§67.9）。加宽度会让 engram.key/val
      跟着线性涨（占 63.7% 参数），「加宽度」混入了「加 Engram」；固定 d 只扫 n_slots
      才能把 Engram 表大小这一维单独拿出来。
    ★`reuse_alive=True` 时**跳过** `build_engram_from_examples`，直接拷 `_ALIVE_CACHE`
      —— 同一份训练数据算出的 alive 是确定的，重复算纯属浪费
      （实测 N=124,593 时 21 s，且在完整流程里被算了两遍）。
    """
    from model import RCfg, EngramRuleLM
    cfg = dict(ck["cfg"])
    cfg["d"] = d
    cfg["n_heads"] = d // HEAD_DIM
    if slots is not None:
        cfg["n_slots"] = int(slots)
    torch.manual_seed(SEED)
    model = EngramRuleLM(RCfg(**cfg)).to(DEV)
    if reuse_alive:
        model.engram.alive.copy_(_ALIVE_CACHE.to(model.engram.alive.device))
    else:
        DS.build_engram_from_examples(model, _TRAIN_X, _TRAIN_L)
    return model


def probe_and_pick_micro(ck, x, starts, lengths, d, slots=None):
    """花 1 步试出跑得动的 micro_batch；单步超时自动降半（§57.4 的闸）。

    ★两条失败判据缺一不可：
      · 抛 OutOfMemoryError
      · **单步超过 PROBE_STEP_TIMEOUT_S 秒** —— 显存打满后的换页抖动**不抛异常**，
        只表现为「极慢、GPU 满载、不返回」（§40 事故的真正症状）。
    """
    mb = micro_batch_for(d)
    last = float("nan")
    while mb >= MIN_MICRO:
        model = make_model(ck, d, slots=slots)
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        train_steps(model, x, starts, lengths, 1, mb, 0.0)
        el = time.time() - t0
        last = el
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
        if el > PROBE_STEP_TIMEOUT_S:
            if mb == MIN_MICRO:
                raise TimeoutError("micro_batch 已到下限 %d，单步仍 %.1fs" % (mb, el))
            print("      d=%d micro_batch=%d 单步 %.1fs > %.1fs ⇒ 降到 %d"
                  % (d, mb, el, PROBE_STEP_TIMEOUT_S, mb // 2), flush=True)
            mb //= 2
            continue
        return mb, el
    raise RuntimeError("micro_batch 降到下限仍跑不动: d=%d（最后单步 %.1fs）" % (d, last))


# ── 断点续训（§69.1）──────────────────────────────────────────────────

def resume_key(cell, micro_batch, slots=None):
    """恢复键 = **全部**自变量。

    ★这是 §67 那个坑的同源防线：`done()` 曾经只按 (n,d,wd) 判完成，加 `--slots`
      后三格里两格被**静默**跳过（日志写着「已完成」，看着像成功）。
      恢复键少一个维度 ⇒ 拿 A 格的权重去接着跑 B 格，比从头跑更糟：**它看起来是成功的**。
    """
    return dict(n=int(cell["n"]), d=int(cell["d"]),
                wd=float(cell["weight_decay"]), steps=int(cell["steps"]),
                micro_batch=int(micro_batch), seq=SEQ, bs=BS, lr=LR,
                seed=SEED, n_slots=int(slots) if slots is not None else 4096)


def resume_path_for(cell, micro_batch, slots=None):
    """恢复文件名**自带全部自变量**：撞名就说明两格配置相同，可以安全复用。"""
    return os.path.join(RESUME_DIR, "resume_v2_n%d_d%d_wd%s_s%d_mb%d_slots%d.pt"
                        % (cell["n"], cell["d"],
                           str(cell["weight_decay"]).replace(".", "p"),
                           cell["steps"], micro_batch,
                           slots if slots is not None else 4096))


def _atomic_save(obj, path):
    """先写 `.tmp` 再 `os.replace`：崩在写盘中途不会留下半个文件。

    ★这个守卫不是洁癖：断点续训**正是为「中途崩掉」准备的**，而没有原子性的话，
      崩掉本身会把**上一个可用恢复点**一起赔进去 —— 正好在最需要它的时候失效。
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_resume(path, key, model, opt, sched, gen, step, trace):
    """落盘恢复点，返回本次存盘的墙钟秒（P5 要拿它算开销占比）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    _atomic_save(dict(key=key,
                      model=model.state_dict(),
                      opt=opt.state_dict(),
                      sched=sched.state_dict(),
                      gen=gen.get_state(),          # ★不存它 ⇒ 样本序列变了，那不是续训
                      torch_rng=torch.get_rng_state(),
                      step=int(step), trace=list(trace),
                      math_gate=MGATE, saved_at=time.time()),
                 path)
    return time.time() - t0


def load_resume(path, key):
    """读恢复点并**逐项核对恢复键**；任一不符即抛错，绝不「尽力恢复」。"""
    st = torch.load(path, map_location="cpu", weights_only=False)
    got, want = st.get("key"), key
    if not isinstance(got, dict):
        raise RuntimeError("恢复点缺少 key 字段: %s" % path)
    bad = [k for k in sorted(set(want) | set(got)) if got.get(k) != want.get(k)]
    if bad:
        raise RuntimeError(
            "恢复键不匹配，拒绝恢复 %s\n  差异字段: %s\n  期望: %s\n  实际: %s"
            % (os.path.basename(path), bad,
               {k: want.get(k) for k in bad}, {k: got.get(k) for k in bad}))
    if not _mgate_equivalent(st.get("math_gate")):
        # ★2026-10-11：原来这里是 `!=` 严格比较，会把**中性迁移**覆盖的恢复点也一起拒掉
        #   —— 本轮 `dist/dialogue_small.py` 加 think 支持后 MGATE 变了，但出货路径
        #   逐位不变（check_think_backcompat.py 的 B1），那个 15.67% 的 N=124593×d=128
        #   恢复点本该还能续。⇒ 改走与 `gate_ok` 同一套等价判定。
        raise RuntimeError(
            "恢复点的 math_gate=%s 与当前 %s 不等价：训练数值代码已变，恢复点作废"
            % (st.get("math_gate"), MGATE))
    return st


def train_steps(model, x, starts, lengths, steps, micro_batch, weight_decay,
                resume=None):
    """与出货 `train_dialogue` 同一套前向/loss，**只把 weight_decay 提成参数**。

    ★为什么不直接调 `train_dialogue`：它把 `weight_decay=0.01` 写死在函数体里
      （`dialogue_small.py:417`），而 wd 是 v2 的显式因子。为**不动出货路径**
      （规则 3），这里复刻它的 loss 归一化、截断逻辑与 OneCycle 计划；
      微批切分的梯度等价性与 `train_dialogue` docstring 里的论证一致。

    ★断点续训（§69.1）：`resume=(path, key)` 时每 `CKPT_EVERY` 步**原子**存一次恢复点，
      并在启动时从恢复点接续。`resume=None`（探针路径）**完全不写盘**。
      `path` 不存在 ⇒ 当作从头跑（不是错误）。

    ★返回 **`(trace, rinfo)` 二元组**，不是 list。调用方按 list 处理会把整个元组
      塞进落盘记录 —— 与 §67 里 `DS.sample_ranges` 返回二元组是同一个错。
    """
    if steps < 1 or micro_batch < 1 or micro_batch > x.shape[0]:
        raise ValueError("invalid steps/micro_batch")
    if not 0.0 <= weight_decay <= 1.0:
        raise ValueError("weight_decay 必须在 [0,1]")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                               pct_start=0.1)
    model.train()
    gen = torch.Generator().manual_seed(SEED)
    bucket_step = getattr(model.c, "bucket_step", 0)
    # ★§68.7：x/s/l **常驻 GPU**，步内只做设备内 gather。
    #   为什么：H2D 实测占单步 **38.0%**（86.3 ms / 226.9 ms，见 §68.7 的分段计时）。
    #   根因是 x 是 487 MB 的 CPU 张量，每微批 `.to(DEV)` 是**同步阻塞**调用，
    #   拷贝期间 GPU 在空等上一个微批的尾巴。常驻后这一步彻底消失。
    #   ★**精确等价**：同为 int64 gather，无浮点归约 ⇒ 逐位相同（判据见
    #     check_scaling_v2_speed.py 的 test_gpu_resident_bit_identical）。
    #   ★守卫：剩余显存不足就回退原路径（不赌，§57.5 同思路）。
    x_dev = starts_dev = lengths_dev = None
    if DEV == "cuda":
        need = (x.numel() * x.element_size()
                + starts.numel() * starts.element_size()
                + lengths.numel() * lengths.element_size())
        free, _total = torch.cuda.mem_get_info()
        # 留 2 GiB 余量给激活与优化器状态
        if free - need > 2 * (1 << 30):
            x_dev = x.to(DEV)
            starts_dev = starts.to(DEV)
            lengths_dev = lengths.to(DEV)
            print("      [x/s/l 常驻 GPU：%.0f MB，剩余 %.0f MB]"
                  % (need / 1048576.0, (free - need) / 1048576.0), flush=True)
        else:
            print("      [显存不足（需 %.0f MB + 2 GiB 余量，实际剩 %.0f MB）⇒ 回退 H2D]"
                  % (need / 1048576.0, free / 1048576.0), flush=True)
    _x = x_dev if x_dev is not None else x
    _s = starts_dev if starts_dev is not None else starts
    _l = lengths_dev if lengths_dev is not None else lengths

    # ★§69.1 断点续训。恢复发生在**步循环之前**，且必须同时恢复五个有状态对象：
    #   opt（AdamW 动量）/ sched（LR 曲线位置）/ gen（样本序列）/ 全局 RNG / step。
    #   ★少存任何一个都不叫续训：
    #     · 不存 opt  ⇒ 前若干步动量清零，等于偷偷重热启动
    #     · 不存 sched⇒ LR 从峰值重爬，整条 OneCycle 曲线都错
    #     · 不存 gen  ⇒ 第 7 步起采到不同样本，**那不是续训，是换数据重跑**（P2 会抓到）
    rinfo = dict(resumed_from=0, n_saves=0, save_wall_sec=0.0)
    start_step = 0
    trace = []
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
            print("      [续训：已完成 %d/%d 步，trace 已恢复 %d 点]"
                  % (start_step, steps, len(trace)), flush=True)
            if start_step >= steps:
                print("      [恢复点已到总步数 ⇒ 本格训练部分视为完成]", flush=True)
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
            # 固定步长下截断到该批真实最大行长是**精确等价**（见 _trim_length 论证）
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
        # ★周期存盘：`(step+1) % CKPT_EVERY` 覆盖「跑到一半被杀掉」，
        #   `step+1 == steps` 覆盖「最后一步跑完但还没落最终模型就崩」。
        #   ★存点必须在 `sched.step()` **之后** —— RNG 与 LR 都已推进到 step+1 的状态，
        #     否则恢复后第一步会重复消费同一个 lr/样本。
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
    """启动前硬闸（§57.5）：空闲显存不足就拒绝启动，不赌。"""
    if DEV != "cuda":
        return
    free, total = torch.cuda.mem_get_info()
    free_mib = free / (1 << 20)
    if free_mib < MIN_FREE_MIB:
        raise RuntimeError("空闲显存 %.0f MiB < %d MiB，拒绝启动"
                           % (free_mib, MIN_FREE_MIB))
    print("  空闲显存 %.0f MiB / 总 %.0f MiB" % (free_mib, total / (1 << 20)), flush=True)


def run_cell(cell, train_rows, probe_pack, tok, slots=None, resume_on=True):
    """跑一格：先探 micro_batch，再正式训练，最后在**固定探针集**上量 NLL。

    ★`resume_on=False` 用于短程自测（不想污染恢复目录）；正式网格保持 True。
    """
    x, s, l = encode_cached(train_rows, tok, SEQ)
    global _TRAIN_X, _TRAIN_S, _TRAIN_L
    _TRAIN_X, _TRAIN_S, _TRAIN_L = x, s, l

    ck = _CK
    # ★优化 1：alive 只算一次。先建一个模型跑 build_engram，把 alive 缓存下来；
    #   之后 mb 探测与正式训练都复用（实测 N=124,593 时省 21 s，原来算了两遍）。
    global _ALIVE_CACHE
    _m0 = make_model(ck, cell["d"], slots=slots)          # 这一趟算 alive
    _ALIVE_CACHE = _m0.engram.alive.detach().clone()
    del _m0
    if DEV == "cuda":
        torch.cuda.empty_cache()
    # ★优化 2：d<=128 时 mb 已知为 64（micro_batch_for 直接返回，不是试出来的），
    #   跳过整段探测（实测省 24 s，其中大头就是重复的 build_engram）。
    #   只对 d>128 才真探测 —— 那里 mb 可能撞显存墙，必须试。
    if cell["d"] <= 128:
        mb, probe_sec = micro_batch_for(cell["d"]), float("nan")
    else:
        mb, probe_sec = probe_and_pick_micro(ck, x, s, l, cell["d"], slots=slots)
    model = make_model(ck, cell["d"], slots=slots, reuse_alive=True)
    params = sum(p.numel() for p in model.parameters())
    engram = sum(p.numel() for n_, p in model.named_parameters()
                 if n_.startswith("engram"))
    px, ps, pl = probe_pack
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # ★§69.1：恢复路径**由 mb 决定**（mb 变了 ⇒ 恢复键不同 ⇒ 必须从头跑）。
    #   resume_on=False 时传 None，`train_steps` 完全不写盘（P4：探针/自测不许污染恢复目录）。
    resume = None
    if resume_on:
        resume = (resume_path_for(cell, mb, slots), resume_key(cell, mb, slots))
    t0 = time.time()
    trace, rinfo = train_steps(model, x, s, l, cell["steps"], mb,
                               cell["weight_decay"], resume=resume)
    wall = time.time() - t0
    # ★元组陷阱的运行时守卫：§67 里我按 list 处理 `sample_ranges` 的二元组，
    #   落盘时把整个 list 塞进一行 JSON。这里在**写盘之前**就把它挡住。
    if not isinstance(trace, list):
        raise RuntimeError("train_steps 的第一个返回值不是 list（是 %s）⇒ 二元组接错了"
                           % type(trace).__name__)
    pm = probe_nll(model, px, ps, pl)
    peak = (torch.cuda.max_memory_reserved() / (1 << 20)) if DEV == "cuda" else 0.0
    # ★在 `del model` **之前**把 cfg 侧的量取出来：del 之后 model 就不绑定了，
    #   上一版我在这里写 `model.c.n_slots` 直接 UnboundLocalError（跑了一整格才炸）。
    n_slots = int(model.c.n_slots)
    alive_slots = int(model.engram.alive.sum())
    ckpt = os.path.join(HERE, "engram_rule_scaling_v2_n%d_d%d_wd%s.pt"
                        % (cell["n"], cell["d"], str(cell["weight_decay"]).replace(".", "p")))
    torch.save(dict(cfg=model.c.__dict__, state_dict=model.state_dict(),
                    tokenizer=_TOK_META, cell=dict(cell, gate=GATE)), ckpt)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    # ★格正常跑完 ⇒ 删掉恢复点，于是「恢复点存在」⟺「该格未完成」，不留歧义。
    if resume is not None and os.path.exists(resume[0]):
        os.remove(resume[0])

    # ★墙钟闸：单步超**同 N 档**基线 ×3 就判异常，**自行停下**而不是硬跑
    #   （旧版用跨 N 的常数基线，必然误判 —— 见 REF_SEC_PER_STEP_BY_N 的注释）
    sec_per_step = wall / cell["steps"]
    ref = ref_for(cell["n"], cell["d"])
    # ★`cell` 里已经有 micro_batch（那是**目标值**），探测后拿到的实测值必须覆盖它，
    #   所以先拷贝 cell 再写，避免 `dict(**cell, micro_batch=...)` 撞键。
    rec = dict(cell)
    rec.update(gate=GATE, math_gate=MGATE, micro_batch=mb,
               probe_step_sec=round(probe_sec, 3),
               params=params, engram_elements=engram,
               engram_share=engram / max(params, 1),
               probe=pm, wall_sec=round(wall, 1),
               sec_per_step=round(sec_per_step, 3), ref_sec_per_step=ref,
               slow_run=sec_per_step > ref * SLOW_MULTIPLE,
               peak_mib=round(peak, 1), loss_trace=trace, ckpt=ckpt,
               n_slots=n_slots, alive_slots=alive_slots,
               resumed_from=rinfo["resumed_from"], n_saves=rinfo["n_saves"],
               save_wall_sec=round(rinfo["save_wall_sec"], 2))
    return rec


_CK = None
_TOK = None
_TOK_META = None


def load_all():
    """返回 (ck_v11, tokenizer)。cfg 取自 v11，只覆盖 d / n_heads。"""
    from tokenizers import Tokenizer
    global _CK, _TOK, _TOK_META
    v11 = os.path.join(HERE, "archive", "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
    ck = torch.load(v11, map_location="cpu", weights_only=False)
    _CK, _TOK_META = ck, ck["tokenizer"]
    _TOK = Tokenizer.from_str(ck["tokenizer"]["json"])
    return ck, _TOK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default=None, help="逗号分隔的 N 列表")
    ap.add_argument("--ds", default=None, help="逗号分隔的 d 列表")
    ap.add_argument("--wds", default=None, help="逗号分隔的 weight_decay 列表")
    ap.add_argument("--force", action="store_true", help="忽略已完成格，重跑")
    ap.add_argument("--no-floor", action="store_true",
                        help="关掉保底 200 步，让 epochs 严格恒定（默认开保底只为与 v1 可比）")
    ap.add_argument("--epochs", type=float, default=EPOCHS_TARGET,
                        help="epochs 目标；--scale-epochs 时它只是 d=32 的基准值")
    ap.add_argument("--scale-epochs", action="store_true",
                        help="epochs 按 d 同比放大：d=32 用 --epochs，d=64 用 2x，d=128 用 4x（Chinchilla 式配比）")
    ap.add_argument("--slots", type=int, default=None,
                        help="覆盖 n_slots（Engram 表大小）—— 换一个容量维度做 A/B")
    ap.add_argument("--workers", type=int, default=1,
                        help="并行 worker 数（>1 时用多进程；受单卡显存限制，实测 d=64 约 2-3 个）")
    ap.add_argument("--gpu-frac", type=float, default=0.0,
                        help="每 worker 的 per_process_memory_fraction（0=不限制）")
    ap.add_argument("--no-resume", action="store_true",
                    help="关掉断点续训（自测用：不在恢复目录里留文件）")
    a = ap.parse_args()
    print("gate = %s（exp_scaling_grid_v2 + model + 两份 dialogue_small）" % GATE)
    print("math_gate = %s（只含 model.py + 两份 dialogue_small；编排改动不影响它）"
          % MGATE)
    print("旧 gate 等价声明: %s ← %s"
          % (LEGACY_GATE_EQUIV["gates"], LEGACY_GATE_EQUIV["math_gate"]))

    ck, tok = load_all()
    rows = load_rows()
    train_rows, probe_rows = split_rows(rows)
    assert_no_leak(train_rows, probe_rows)
    print("  语料 %d 行 → train %d / 探针 %d（全量词项级不相交断言通过）"
          % (len(rows), len(train_rows), len(probe_rows)), flush=True)

    probe_pack = encode_cached(probe_rows, tok, SEQ)
    ns = tuple(int(v) for v in a.ns.split(",")) if a.ns else N_LIST
    ds = tuple(int(v) for v in a.ds.split(",")) if a.ds else D_LIST
    wds = tuple(float(v) for v in a.wds.split(",")) if a.wds else WD_LIST
    pool = len(train_rows)
    for n in ns:
        if n > pool:
            raise SystemExit("N=%d 超过训练池 %d" % (n, pool))

    floor = 0 if a.no_floor else FLOOR_STEPS
    cells = []
    for d in ds:
        ep = a.epochs * (d / min(ds) if a.scale_epochs else 1.0)
        cells += build_grid(ns, (d,), wds, floor=floor, ep_target=ep)
    cells.sort(key=lambda c: (c["n"], c["d"], c["weight_decay"]))
    print("  网格 %d 格（epochs 目标 %.1f，wd 臂 %s）"
          % (len(cells), EPOCHS_TARGET, list(wds)), flush=True)
    check_free_memory()

    def done(cell):
        """★完成判定必须包含 `n_slots` 与 `steps`：否则 slots=256 跑完后,
        slots=1024/4096 会被判成「已完成」而跳过（这次就踩了 ——
        三格只跑了一格，日志里后两格全是「跳过（已完成）」）。

        ★§69.3：`gate` 换成 `gate_ok(j)`。它对**新记录**比 `math_gate`（只覆盖
          能改训练数值的文件），对**旧记录**走显式等价声明。
          编排改动（断点续训、预取）不再作废已有读数；真改 model.py 仍全部作废。
        """
        if not os.path.exists(OUT) or a.force:
            return False
        want_slots = a.slots if a.slots is not None else 4096
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                if (gate_ok(j) and j.get("n") == cell["n"]
                        and j.get("d") == cell["d"]
                        and j.get("steps") == cell["steps"]
                        and j.get("weight_decay") == cell["weight_decay"]
                        and j.get("n_slots") == want_slots):
                    return True
        return False

    # ★预计墙钟用**同 N 档**基线，不再用跨 N 的常数（旧版在这里把 d=64/N=31,148
    #   估成几百秒，实测 1106 s，三格合计超时被砍 —— 见 §67.10 第 5 条）
    est = sum(c["steps"] * ref_for(c["n"], c["d"]) for c in cells)
    print("  预计总墙钟 ~%.0f s（%.1f min，按同 N 档实测基线外推）"
          % (est, est / 60.0), flush=True)

    for cell in cells:
        if done(cell):
            print("  跳过 N=%-5d d=%-4d wd=%-5.2f（已完成）"
                  % (cell["n"], cell["d"], cell["weight_decay"]), flush=True)
            continue
        print("  >>> N=%-5d d=%-4d wd=%-5.2f steps=%-5d epochs=%.1f ..."
              % (cell["n"], cell["d"], cell["weight_decay"],
                 cell["steps"], cell["epochs"]), flush=True)
        sub = train_rows[:cell["n"]]
        rec = run_cell(cell, sub, probe_pack, tok, slots=a.slots,
                       resume_on=not a.no_resume)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("      probe NLL=%.4f tok_acc=%.4f | train末loss=%.4f | "
              "mb=%d %.1fs (%.3f s/步, 基线 %.3f)%s"
              % (rec["probe"]["probe_nll"], rec["probe"]["probe_tok_acc"],
                 rec["loss_trace"][-1], rec["micro_batch"], rec["wall_sec"],
                 rec["sec_per_step"], rec["ref_sec_per_step"],
                 "  ★SLOW 已标记" if rec["slow_run"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
