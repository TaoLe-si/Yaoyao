"""★预登记: 检验"绑定失败是因为窗口回看长度不够"这个假设。

────────────────────────────────────────────────────────────────────────
**先纠正一个方法论问题**(否则会得到不可判的结论)

外部建议提出"在 v11 上做推理期窗口扫描 W=16→32→64, 看首 token acc 是否改善", 并推论
"若几乎不变 ⇒ 模型没学会用长窗口, 重训也未必有用"。

**这个否定分支不成立。** W 是模型**没见过的**取值:
  ① 局部归一化的分母是 `Σ_w |gate| + ε`, **随 W 改变** ⇒ 输出尺度整体漂移;
  ② `bidx = off // 32` 在 W=32/64 时会跨到 1~3 个桶块 ⇒ `bsel` 的空间结构也变了。
冻结权重在 W≠16 上是**双重离分布**(多了信息 + 丢了标定)。
⇒ **W 扫描只���"正分支"有信息量**:涨了 ⇒ 强证据;没涨 ⇒ **不能**推出重训也无效,
因为"标定漂移"完全可以掩盖"信息变多"。本脚本据此只把正分支当证据。

────────────────────────────────────────────────────────────────────────
**所以先做在分布(A)、无架构改动的测量(B)** —— 它才是判决性的

────────────────────────────────────────────────────────────────────────
**预登记预测** (写在跑之前)

A1 (★判决性, 在分布, 无架构改动) 把 train 820 按**问题 token 长度**分层,
   分界取 **15 = 窗口回看上限**:
     若 acc(q≤15) 明显高于 acc(q>15) ⇒ **覆盖不足是绑定失败的真实原因**。
     若几乎持平 ⇒ 覆盖不是原因, **B 的正分支也不该期待**, 窗口扫描可以降级。
   附带更细的分位分层看是否单调。
   ⚠ 已知混淆: 长问题本身可能更难(与覆盖无关)。因此只把它当**必要条件筛查**,
   不把"持平"直接读成否证, 也不把"分层差异"直接读成确证。

B2 (离分布, 只看正分支) W=16/32/64 的首 token acc。
   若 W=32 > W=16 ⇒ 长窗口确实带来可用信息, 值得预登记重训实验。
   若不升 ⇒ **不下结论**(见上), 只记"该读数无信息量"。

**诚实边界**: v11 单 seed, 无种子极差。A 是相关不是因果; 覆盖与难度混淆未消除。
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
SEQ = 512
USER_ID = 4
# 与 train.py:17 同一口径; 本探针若留在 CPU 上会慢到不可用 (实测 5 遍 820x512 跑不完)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# AGENTS.md 规则 18 的 bs≥256 针对**训练**(避免 launch-bound)。本探针是推理:
# SwiGLU 门产生 (B,L,W,H,Dg) 中间张量, 大小 ∝ B·W。bs=256/W=16 时单个已约 537MB,
# 多项叠加把显存打满 => 退化成换页抖动, 实测单批 >30s。bs=32 时实测 32.4ms/批
# (50 万 token/s), 已远非 launch-bound, 故此处取 32 而非 256。
# 规则 20 的 profile 读数见 bench_gpu.py。
BS = 32
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 只跑前 N 条(0=全部); 调试用


def rebuild(ck):
    from tokenizers import Tokenizer
    from dialogue_small import split_records, encode_examples
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr, ev, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                              seed=29, return_meta=True)
    return encode_examples(tr, tok, SEQ), encode_examples(ev, tok, SEQ)


def qlen_of(x, starts):
    """问题正文 token 数 (从 [USER] 之后到答案首 token 之前)。"""
    out = []
    for i in range(x.shape[0]):
        pre = x[i, :int(starts[i])].tolist()
        u = max([j for j, t in enumerate(pre) if t == USER_ID], default=0)
        out.append(int(starts[i]) - u - 1)
    return torch.tensor(out)


@torch.no_grad()
def first_token(logits, x, starts):
    bi = torch.arange(x.shape[0])
    tgt = x[bi, starts]
    lg = logits[bi, starts - 1]
    return (lg.argmax(-1) == tgt).float(), F.cross_entropy(lg, tgt, reduction="none")


def acc_for(model, x, s):
    """首 token 命中向量 (CPU)。模型与输入都必须上 DEV。

    分块按 W 自适应: SwiGLU 门产生 (B,L,W,H,Dg) 中间张量, 单个在 bs=256/W=16 时
    已约 537MB, W 翻倍则翻 4 倍 —— 8GB 卡上必须让 B·W 近似恒定, 否则显存打满后
    退化成换页抖动(表现为"在跑但极慢")。
    返回 (hits, bs, seconds)。
    """
    W = model.c.window_max
    bs = max(16, BS // max(1, W // 16))
    n = x.shape[0] if LIMIT <= 0 else min(LIMIT, x.shape[0])
    hits, t0 = [], time.time()
    for i in range(0, n, bs):
        tb = time.time()
        xb = x[i:i + bs].to(DEV)
        sb = s[i:i + bs].to(DEV)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t_h2d = time.time()
        lg = model(xb).float()
        if DEV == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.time()
        bi = torch.arange(xb.shape[0], device=DEV)
        hits.append((lg[bi, sb - 1].argmax(-1) == xb[bi, sb]).float().cpu())
        if DEV == "cuda":
            torch.cuda.synchronize()
        print("      batch %3d/%d n=%d  h2d=%.0fms fwd=%.0fms"
              % (i // bs + 1, (n + bs - 1) // bs, xb.shape[0],
                 (t_h2d - tb) * 1e3, (t_fwd - t_h2d) * 1e3))
        del xb, sb, lg, bi
    return torch.cat(hits), bs, time.time() - t0


def main():
    from model import RCfg, EngramRuleLM

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    (tr_x, tr_s, tr_l), (ev_x, ev_s, ev_l) = rebuild(ck)
    print("DEV = %s | torch %s | LIMIT=%d" % (DEV, torch.__version__, LIMIT))

    # S1: 重建必须与 checkpoint 自带 eval 张量逐元素相同, 否则全部读数作废
    same = bool(ev_x.shape == ck["eval_tokens"].shape) and bool((ev_x == ck["eval_tokens"]).all())
    print("[S1] eval 重建逐元素相同 =", same)
    if not same:
        print("★自检失败, 中止。")
        return 2
    print("[S1] train=%d eval=%d  → 自检通过" % (tr_x.shape[0], ev_x.shape[0]))

    base_cfg = ck["cfg"]
    qlen = qlen_of(tr_x, tr_s)
    print("\n[A] 问题长度分层 (train 820), 分界 15 = 窗口回看上限")
    print("    全样本: qlen 中位 %.0f 均值 %.1f" % (float(qlen.median()), float(qlen.float().mean())))

    model = EngramRuleLM(RCfg(**base_cfg)).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    hit, bs0, sec0 = acc_for(model, tr_x, tr_s)
    print("    [基线 W=16] bs=%d 用时 %.1fs" % (bs0, sec0))

    lo = qlen <= 15
    print("    q<=15 (窗口可完整覆盖): n=%3d  首token acc = %.4f" % (int(lo.sum()), float(hit[lo].mean())))
    print("    q>15 (窗口只能看尾部): n=%3d  首token acc = %.4f" % (int((~lo).sum()), float(hit[~lo].mean())))
    gap = float(hit[lo].mean() - hit[~lo].mean())
    print("    ★A1 差值 = %+.4f  %s" % (gap, "命中预测(覆盖不足是真实原因)" if gap > 0.05
                                        else "接近持平(覆盖不是主因)"))

    print("\n    更细分层(五分位)看单调性:")
    qs = torch.quantile(qlen.float(), torch.tensor([0., .2, .4, .6, .8, 1.]))
    for k in range(5):
        a, b = float(qs[k]), float(qs[k + 1])
        m = (qlen >= a) & (qlen <= b if k == 4 else qlen < b)
        if int(m.sum()) == 0:
            continue
        print("      qlen [%3.0f,%3.0f]  n=%3d  acc=%.4f" % (a, b, int(m.sum()), float(hit[m].mean())))

    # ---- B: 推理期窗口扫描 (只认正分支) ----
    print("\n[B] 推理期窗口扫描 W=16/32/64 —— 只认**正分支**")
    print("    (W 未在训练中见过; 局部归一化分母随 W 变、bidx 会跨桶块 ⇒ 双重离分布)")
    for W in (16, 32, 64):
        cfg2 = RCfg(**{**base_cfg, "window_max": W})
        m2 = EngramRuleLM(cfg2).to(DEV)
        m2.load_state_dict(ck["state_dict"])
        m2.eval()
        h, bsw, secw = acc_for(m2, tr_x, tr_s)
        print("    W=%-3d  train 首token acc = %.4f   (bs=%d, %.1fs)" % (W, float(h.mean()), bsw, secw))
        if W == 16:
            same16 = abs(float(h.mean()) - float(hit.mean())) < 1e-6
            print("      └ 与 [A] 基线一致性自检: %s (%.6f vs %.6f)"
                  % ("通过" if same16 else "★失败", float(h.mean()), float(hit.mean())))
            if not same16:
                print("      ★W=16 都复现不出基线, 后面 W 的读数不可信, 中止。")
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())