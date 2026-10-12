"""★ 架构图的**事实来源**: 从真实 checkpoint + 一次真实前向里读出图上每一个数字。

图 `draw_arch_figure.py` 只画, 不算 —— 这样图上任何一处数字都能回到本脚本的落盘 JSON,
而不是画图脚本里手打的常量。**规则: 不猜。**

产出 `arch_figure_facts.json`:
  · 配置(cfg 原样)
  · 参数清单(逐张量: 名字 / 形状 / 元素数 / 类别 标量|向量|矩阵)
  · 派生量(R = ngram_max×n_hash、静态回看半径)
  · 一次真实前向的诊断读数(start_bucket 分布 / 窗口偏移 / 门值 / 存活槽 / 跳过率)

口径: 出货基准 `engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st400.pt`
      (d=128 / n_layers=2 / seed=11 / 400 步)。与 DESIGN §59/§64 同一把权重。
"""
import hashlib
import json
import os
import sys
from collections import Counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CKPT = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st400.pt")
V11 = os.path.join(HERE, "archive", "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
OUT = os.path.join(HERE, "arch_figure_facts.json")
SEQ = 512
N_FWD = 32           # 前向的批大小(诊断读数用, 不是评测)


def categorize(t):
    """按维数分类。规则 7 精神: 分类本身要可失败 —— 维数决定类别, 无例外分支。"""
    if t.dim() == 0:
        return "scalar"
    if t.dim() == 1:
        return "vector"
    return "matrix"


def gate_tag():
    h = hashlib.sha256()
    # ★主目录的 dialogue_small.py **import 不了**(它第 20 行 `import train`,
    #   而 train.py 已归档、archive 下又没有它要的 baseline 包)。实际加载的是
    #   `dist/dialogue_small.py` —— 与主目录**逐字相同、只少 3 行 import**
    #   (requests / train_module / from train import), 已用 difflib 核过。
    #   所以门禁指纹覆盖这两个文件 + model.py。
    for name in ("probe_arch_figure.py", "model.py",
                 "dialogue_small.py", os.path.join("dist", "dialogue_small.py")):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:12]


def load_dialogue_small():
    """按路径显式加载 `dist/dialogue_small.py`, 模块名 `ds_offline`。

    为什么不能用普通的 `import dialogue_small`:
      · 主目录那份第 20-21 行 `import train as train_module` / `from train import ...`,
        而 `train.py` 已归档进 `archive/`, 且 `archive/train.py` 自己要
        `from baseline.dense import DenseLM` —— archive 下**没有** baseline 包
        ⇒ 无论从哪个目录起都 import 不了。
      · `dist/dialogue_small.py` 与主目录**逐字相同**, 只少那 3 行 import
        (requests / train_module / from train import), 已 diff 核实。
    sys.path 里主目录在首位 ⇒ 它内部 `from model import ...` 拿到的是**主目录的
    model.py**, 不会误用 dist 的副本。
    """
    import importlib.util
    path = os.path.join(HERE, "dist", "dialogue_small.py")
    spec = importlib.util.spec_from_file_location("ds_offline", path)
    assert spec is not None and spec.loader is not None, "spec 构造失败: %s" % path
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ds_offline"] = mod
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def main():
    gate = gate_tag()
    from model import RCfg, EngramRuleLM

    if not os.path.exists(CKPT):
        raise SystemExit("找不到基准 checkpoint: %s" % CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg_raw = dict(ck["cfg"])
    sd = ck["state_dict"]

    # ---------- 2. 派生量 ----------
    c = RCfg(**cfg_raw)
    R = c.n_cand                       # = ngram_max × n_hash
    static_radius = (c.n_buckets - 1) * c.bucket_step + c.window_max - 1
    window_offsets = c.window_max      # 窗口里有多少个偏移

    # ---------- 3. 构造 + 载入 ----------
    torch.manual_seed(11)
    model = EngramRuleLM(c)
    model.load_state_dict(sd, strict=True)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    # ---------- 1. 参数清单 ----------
    # ★口径澄清(我先前口头报的 1,651,258 是错的): state_dict 有 **63** 个张量, 但其中
    #   3 个是 `register_buffer` 的**非可训练**哈希预计算表(h_mult / h_mod / alive),
    #   合计 4,108 元素。**可训练参数 = 1,647,150** —— 与 train_budget_v1.jsonl 里
    #   `params` 字段(exp_train_budget 用 model.parameters() 求和)同口径。
    #   图上标 1,647,150; 1,651,258 只作"含 buffer 的张量元素总数"出现, 且必须标明。
    trainable = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    params, by_cat = [], Counter()
    cat_numel = Counter()
    for name, t in sorted(trainable.items()):
        cat = categorize(t)
        n = int(t.numel())
        params.append(dict(name=name, shape=list(t.shape), numel=n, cat=cat))
        by_cat[cat] += 1
        cat_numel[cat] += n
    trainable_elem = sum(p["numel"] for p in params)

    buf_list = [dict(name=n, shape=list(t.shape), numel=int(t.numel()),
                     dtype=str(t.dtype).replace("torch.", ""))
                for n, t in sorted(buffers.items())]
    buf_elem = sum(b["numel"] for b in buf_list)
    sd_elem = sum(int(sd[k].numel()) for k in sd)

    # ★自洽锚点(规则 23: 检查的锚点必须与目标口径一致)
    assert cat_numel["scalar"] + cat_numel["vector"] + cat_numel["matrix"] == trainable_elem
    assert trainable_elem == sum(int(t.numel()) for t in trainable.values()), "参数求和不自洽"
    assert set(trainable) | set(buffers) == set(sd), \
        "可训练张量 + buffer 与 state_dict 的键集合不一致"
    assert trainable_elem + buf_elem == sd_elem, \
        "可训练 %d + buffer %d != state_dict %d" % (trainable_elem, buf_elem, sd_elem)
    assert trainable_elem == 1647150, \
        "可训练参数 %d != 落盘记录 train_budget_v1.jsonl 的 params=1647150" % trainable_elem
    assert len(params) == len(sd) - len(buffers), "张量个数不自洽"

    # ---------- 4. 真实前向 ----------
    from tokenizers import Tokenizer
    ds = load_dialogue_small()
    v11 = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(v11["tokenizer"]["json"])
    tr_rows, _ev, _meta = ds.split_records(v11["extra"]["raw_dialogue_rows"],
                                          return_meta=True)
    # ★口径: split_records 的默认参数就是 train_fraction=0.8 / seed=29,
    #   与训练时 exp_capacity_scaling.load_data 的 train_frac=0.8 / seed=29 同值。
    #   (归档的 exp_capacity_scaling 写的是 train_frac=, 现版 dialogue_small 改名为
    #    train_fraction=, 调用方按默认值走即可, 同一个函数同一个种子。)
    x, s, l = ds.encode_examples(tr_rows[:N_FWD], tok, SEQ)
    L_used = int(l.max())
    x = x[:, :max(L_used, 2)]
    xd = x.to(dev)

    for blk in model.blocks:
        blk.attn.record = True
        blk.attn.record_gc = True
    with torch.no_grad():
        logits = model(xd)
    assert logits.shape == (x.shape[0], x.shape[1], c.vocab), \
        "前向形状 %s 与预期不符" % (tuple(logits.shape),)

    att0 = model.blocks[0].attn
    attL = model.blocks[-1].attn

    # start_bucket 分布 —— 只对"规则路径没跳过"的位置统计(跳过位偏置恒 0 ⇒ argmax 恒 0)
    sb = att0.last_startb.float()                      # (B,L)
    off = att0.last_window_off                          # (B,L,W) 夹紧后的实际偏移
    off_flat = off.reshape(-1, off.shape[-1])

    diag = dict(
        n_forward_rows=int(x.shape[0]),
        seq_len_used=L_used,
        rule_skip_ratio=model.stats()["rule_skip_ratio"],
        alive_slots=int(model.engram.alive.sum()),
        n_slots=int(c.n_slots),
        alive_ratio=float(model.engram.alive.float().mean()),
        start_bucket_layer0=dict(
            distinct=sorted({int(v) for v in sb.unique().tolist()}),
            histogram={int(k): int(v) for k, v in
                       sorted(Counter(sb.flatten().tolist()).items())},
        ),
        window=dict(
            W=int(off.shape[-1]),
            offset_min=int(off_flat.min().item()),
            offset_max=int(off_flat.max().item()),
            offset_mean=float(off_flat.float().mean().item()),
        ),
        scalars={},
        per_layer=[],
    )

    # 标量参数的真实取值 —— 图上要写数
    for tag, obj in (("layer0", att0), ("layerL", attL)):
        mag = dict(obj.last_mag)
        diag["scalars"][tag] = dict(
            rule_scale=float(obj.rule_scale.item()),
            lam=float(obj.lam.item()),
            head_w_mean=float(mag["head_w_mean"]),
            gate_mean=float(mag["gate_mean"]),
            bias_absmean=float(mag["rule_absmean"]),
            content_gate=float(obj.last_gc),
        )
    for i, blk in enumerate(model.blocks):
        o = blk.attn
        diag["per_layer"].append(dict(
            layer=i,
            rule_scale=float(o.rule_scale.item()),
            lam=float(o.lam.item()),
            head_w_mean=float(o.head_w.mean().item()),
            gate_mean=float(o.last_mag["gate_mean"]),
            content_gate=float(o.last_gc),
        ))

    # ---------- 5. Engram 读的真实候选结构 ----------
    slot = model.engram.slot_ids(xd)                    # (B,L,R)
    flat = slot.reshape(-1, slot.shape[-1])
    # ★每个位置候选槽的去重个数: 用**排序 + 相邻不等计数**向量化, 不用逐行 unique()
    #   (B·L 次小 unique 在 CUDA 上要几秒, 而且会掩盖真实读数)
    srt, _ = torch.sort(flat, dim=1)
    distinct = 1 + (srt[:, 1:] != srt[:, :-1]).sum(1) if srt.shape[1] > 1 \
        else torch.ones(srt.shape[0], dtype=torch.long, device=srt.device)
    diag["engram"] = dict(
        n_cand=int(slot.shape[-1]),
        mean_distinct_slots_per_pos=float(distinct.float().mean().item()),
        min_distinct_slots_per_pos=int(distinct.min().item()),
        max_distinct_slots_per_pos=int(distinct.max().item()),
        live_frac_per_pos=float(model.engram.alive[slot].float().mean().item()),
    )

    # ---------- 6. 参数构成按模块 ----------
    mod_numel = Counter()
    for p in params:
        top = p["name"].split(".")[0]
        mod_numel[top] += p["numel"]

    facts = dict(
        gate=gate,
        ckpt=os.path.basename(CKPT),
        ckpt_bytes=os.path.getsize(CKPT),
        extra=ck.get("extra", {}),
        cfg=cfg_raw,
        derived=dict(
            n_cand=int(R),
            head_dim=int(c.head_dim),
            static_lookback_radius=int(static_radius),
            window_offsets=int(window_offsets),
            bucket_step=int(c.bucket_step),
            max_window_offset_per_bucket=int(c.bucket_step * (c.n_buckets - 1)),
        ),
        params=dict(
            list=params,
            n_tensors=len(params),
            trainable_elements=trainable_elem,
            by_cat_count=dict(by_cat),
            by_cat_numel=dict(cat_numel),
            by_module_numel=dict(mod_numel),
            engram_share=mod_numel["engram"] / max(trainable_elem, 1),
            buffers=buf_list,
            buffer_elements=buf_elem,
            state_dict_tensors=len(sd),
            state_dict_elements=sd_elem,
        ),
        diag=diag,
    )

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)

    print("gate = %s" % gate)
    print("ckpt = %s (%s bytes)"
          % (facts["ckpt"], "{:,}".format(facts["ckpt_bytes"])))
    print("可训练参数张量 %d 个 / %s 元素"
          % (len(params), "{:,}".format(trainable_elem)))
    for cat in ("scalar", "vector", "matrix"):
        print("  %-6s %d 个 / %s 元素"
              % (cat, by_cat[cat], "{:,}".format(cat_numel[cat])))
    print("非训练 buffer %d 个 / %s 元素 (h_mult / h_mod / alive)"
          % (len(buffers), "{:,}".format(buf_elem)))
    print("state_dict 合计 %d 个张量 / %s 元素"
          % (len(sd), "{:,}".format(sd_elem)))
    print("R = %d, 静态回看半径 = %d" % (R, static_radius))
    print("start_bucket(layer0) 取值 = %s" % diag["start_bucket_layer0"]["distinct"])
    print("窗口偏移 [%d, %d]" % (diag["window"]["offset_min"], diag["window"]["offset_max"]))
    print("存活槽 %d / %d (%.4f), 规则跳过率 %.4f"
          % (diag["alive_slots"], diag["n_slots"], diag["alive_ratio"],
             diag["rule_skip_ratio"]))
    print("每位置候选去重槽均值 %.3f" % diag["engram"]["mean_distinct_slots_per_pos"])
    print("已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())