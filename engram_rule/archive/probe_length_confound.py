"""★预登记: 分离"问题长度效应"里的【难度】与【绑定】两个成分。

上一轮实测(§40): 首 token 准确率随问题长度单调递减 0.6753 -> 0.4146, 但
**不能归因给窗口覆盖**, 因为长度与覆盖混淆。本脚本做分离。

────────────────────────────────────────────────────────────────────────
**要区分的两件事**
  H_difficulty  长问题本身更难(答案更长/更多样/更少重复) ⇒ 与窗口无关, 窗口方向放下
  H_binding     长度专门损害"绑定"(首 token) ⇒ 才值得重训 window_max=32

────────────────────────────────────────────────────────────────────────
**预登记三条测量**(外部建议给的两条 + 一条它没提但必须有的)

  M1  按问题长度分层, 测 teacher-forced **答案内部** NLL(不含首 token)
  M2  按问题长度分层, 测 teacher-forced **首 token** NLL
  M3  **控制答案长度**: 在同一答案长度区间内, 比较长/短问题

⚠ M1 的固有漏洞(外部建议已指出, 此处照记): 内部 NLL 偏高可能只因长问题的答案
  本身更长/更复杂, 与覆盖无关。**M1 单独不能证明"长度是难度"**, 只能排除
  "长度专门影响绑定"。M3 就是为了堵这个漏洞。

────────────────────────────────────────────────────────────────────────
**预登记判决规则**(跑之前写死, 跑完只对照)

  R1  M1 与 M2 都随问题长度单调上升 ⇒ 长度是难度, **窗口方向彻底放下**
  R2  M1 不上升但 M2 上升 ⇒ 长度专门影响绑定, 重训 W=32 值得试
  R3  M1、M2 都不上升, 但自由生成首 token 仍错 ⇒ 问题在绑定之后的传播, 不在窗口
  ⚠ 三条都不命中 ⇒ 说明分层本身不解释失败, 回到 §41 的 Δ 诊断

────────────────────────────────────────────────────────────────────────
**S2 实现自检(外部建议没提, 但必须有)**
  本脚本算出的 full 模型全体读数必须复现已公布的 v11 值:
    首 token acc 0.552439 / 答案内部 acc 0.790457 / 答案内部 NLL 1.194452
  (DESIGN §38)。对不上 ⇒ 我的指标实现与项目口径不一致, 全部读数作废。
  这比任何自造判据都硬: 它把实现锚在**已被独立复现过的文档读数**上。

**诚实边界**: v11 单 seed, 无种子极差。M3 的答案长度区间内样本量会变小,
  窄区间可能不足以定论, 该区间会报 n 值。
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
SEQ = 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# 推理期 bs 取 32: 中间张量 ∝ B·W, bs=256 会打满显存退化(见 §40 性能事故)。
# 规则 18 的 bs>=256 针对训练; 此处 32 已达 ~50 万 token/s, 非 launch-bound。
BS = 32
USER_ID = 4

# §38 已公布并被独立复现的 v11 全体读数(S2 自检的锚)。
# ⚠ 口径提示: §38 的 "answer token accuracy / answer NLL" 是**含首 token**的整段答案口径
#   (39,214 个答案 token = 820 首 token + 38,394 内部 token)。本脚本同时需要"不含首 token"
#   的内部口径(M1/M2 的分层对象), 所以 S2 必须锚在**合并口径**上, 否则会把口径差
#   误读成实现 bug —— 第一次跑就是这么误判的。
ANCHOR = dict(answer_acc=0.790457, answer_nll=1.194452, answer_tokens=39214)
TOL = 2e-3          # 容差: §38 自报复现误差 < 2e-6; 这里放宽到 2e-3 吸收实现差异


def rebuild(ck):
    from tokenizers import Tokenizer
    from dialogue_small import split_records, encode_examples
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr, ev, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                              seed=29, return_meta=True)
    return encode_examples(tr, tok, SEQ), encode_examples(ev, tok, SEQ)


def qlen_of(x, starts):
    out = []
    for i in range(x.shape[0]):
        pre = x[i, :int(starts[i])].tolist()
        u = max([j for j, t in enumerate(pre) if t == USER_ID], default=0)
        out.append(int(starts[i]) - u - 1)
    return torch.tensor(out)


@torch.no_grad()
def collect(model, x, starts, lengths):
    """返回 (first_nll, first_acc, internal_nll_sum, internal_cnt, internal_acc_sum)"""
    fn, fa, isum, icnt, iacc = [], [], [], [], []
    for i in range(0, x.shape[0], BS):
        xb = x[i:i + BS].to(DEV)
        sb = starts[i:i + BS].to(DEV)
        lb = lengths[i:i + BS].to(DEV)
        lg = model(xb).float()
        bi = torch.arange(xb.shape[0], device=DEV)
        fn.append(F.cross_entropy(lg[bi, sb - 1], xb[bi, sb], reduction="none").cpu())
        fa.append((lg[bi, sb - 1].argmax(-1) == xb[bi, sb]).float().cpu())
        # logits 索引 p 预测 x[p+1]; 答案内部 token 是 x[start+1 .. length-1]
        p = torch.arange(SEQ - 1, device=DEV).view(1, -1)
        m = (p >= sb.view(-1, 1)) & (p < (lb - 1).view(-1, 1)).clamp_min(0)
        ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                             xb[:, 1:].reshape(-1), reduction="none").view_as(xb[:, 1:])
        isum.append((ce * m).sum(1).cpu())
        icnt.append(m.sum(1).cpu())
        hit = (lg[:, :-1].argmax(-1) == xb[:, 1:]).float()
        iacc.append((hit * m).sum(1).cpu())
    return torch.cat(fn), torch.cat(fa), torch.cat(isum), torch.cat(icnt), torch.cat(iacc)


def mean_in(mask, sumvec, cnt):
    """sumvec/cnt 必须同语义: sumvec 是**已按 token 求和**的量, cnt 是该行的 token 数。
    平均 = Σsumvec / Σcnt。⚠ 不要再乘一遍 cnt —— §40 那次就是这样把 rest_nll 算成 99.5 的,
    被 S2 自检(锚在 §38 已公布读数上)当场抓住。"""
    den = float(cnt[mask].sum())
    return float("nan") if den <= 0 else float(sumvec[mask].sum()) / den


def main():
    from model import RCfg, EngramRuleLM
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    (tr_x, tr_s, tr_l), (ev_x, ev_s, ev_l) = rebuild(ck)
    same = ev_x.shape == ck["eval_tokens"].shape and bool((ev_x == ck["eval_tokens"]).all())
    print("[S1] eval 重建逐元素相同 =", same)
    if not same:
        print("★中止: 重建不一致。")
        return 2

    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    t0 = time.time()
    fn, fa, isum, icnt, iacc = collect(model, tr_x, tr_s, tr_l)
    print("train 前向 %.1fs | DEV=%s" % (time.time() - t0, DEV))

    qlen = qlen_of(tr_x, tr_s)
    alen = tr_l - tr_s
    allm = torch.ones(fn.shape[0], dtype=torch.bool)
    ones = torch.ones_like(fn)      # 首 token 每行恰好 1 个, 权重视作 1

    # ---- S2 实现自检: 合并口径(含首 token)必须锚在 §38 已公布值上 ----
    n_tok = float(fn.numel()) + float(icnt.sum())          # 820 首 + 内部 token 数
    comb = dict(answer_acc=(float(fa.sum()) + float(iacc.sum())) / n_tok,
                answer_nll=(float(fn.sum()) + float(isum.sum())) / n_tok,
                answer_tokens=n_tok)
    print("[S2] 实现自检 (对照 DESIGN §38 已独立复现的 v11 读数, 合并口径含首 token):")
    fails = []
    for k in ("answer_acc", "answer_nll", "answer_tokens"):
        d = comb[k] - ANCHOR[k]
        ok = abs(d) <= TOL
        if not ok:
            fails.append(k)
        print("     %-14s 实测 %12.6f  §38 %12.6f  Δ=%+.2e  %s"
              % (k, comb[k], ANCHOR[k], d, "通过" if ok else "★失败"))
    print("     拆分口径: 首token acc=%.6f | 内部 acc=%.6f | 首token NLL=%.4f | 内部 NLL=%.4f"
          % (float(fa.mean()), mean_in(allm, iacc, icnt), float(fn.mean()), mean_in(allm, isum, icnt)))
    if fails:
        print("★中止: %s 与已公布读数不符。" % ",".join(fails))
        return 2
    print("     → 通过, 实现已锚定。下面所有分层读数继承该口径。")

    print("\n全体: qlen 中位 %.0f | alen 中位 %.0f | alen 范围 [%d, %d]"
          % (float(qlen.median()), float(alen.median()), int(alen.min()), int(alen.max())))

    # ---- M1 / M2: 按问题长度分层 ----
    edges = [0, 12, 17, 21, 27, 10 ** 6]
    print("\n[M1/M2] 按问题长度分层 (train 820)")
    print("  %-14s %5s %10s %10s" % ("qlen 区间", "n", "首token NLL", "内部 NLL"))
    rows = []
    for k in range(len(edges) - 1):
        m = (qlen >= edges[k]) & (qlen < edges[k + 1])
        n = int(m.sum())
        if n == 0:
            continue
        f = mean_in(m, fn, ones)
        r = mean_in(m, isum, icnt)
        rows.append(((edges[k], edges[k + 1]), n, f, r))
        print("  [%3d,%4d)   %5d %10.4f %10.4f" % (edges[k], edges[k + 1], n, f, r))
    inc_f = rows[-1][2] > rows[0][2]
    inc_r = rows[-1][3] > rows[0][3]
    print("  → M2 首token NLL 随 qlen %s | M1 内部 NLL 随 qlen %s"
          % ("上升" if inc_f else "不升", "上升" if inc_r else "不升"))

    # ---- M3: 控制答案长度, 在同一 alen 区间内比较长/短问题 ----
    print("\n[M3] 控制答案长度: 在同一 alen 区间内, 比较 qlen<=15 与 qlen>15")
    print("  %-14s %5s %12s %12s %12s %12s" % ("alen 区间", "n", "短q首NLL", "长q首NLL", "短q内部NLL", "长q内部NLL"))
    aedges = [0, 20, 30, 40, 60, 10 ** 6]
    gaps_f, gaps_r = [], []
    for k in range(len(aedges) - 1):
        band = (alen >= aedges[k]) & (alen < aedges[k + 1])
        s, l = band & (qlen <= 15), band & (qlen > 15)
        ns, nl = int(s.sum()), int(l.sum())
        if ns < 5 or nl < 5:
            print("  [%3d,%4d)   %5d  —— 样本不足(短%d/长%d), 跳过" % (aedges[k], aedges[k + 1], ns + nl, ns, nl))
            continue
        sf, lf = float(fn[s].mean()), float(fn[l].mean())
        sr, lr = mean_in(s, isum, icnt), mean_in(l, isum, icnt)
        gaps_f.append(lf - sf)
        gaps_r.append(lr - sr)
        print("  [%3d,%4d)   %5d %12.4f %12.4f %12.4f %12.4f"
              % (aedges[k], aedges[k + 1], ns + nl, sf, lf, sr, lr))
    if gaps_f:
        print("  → 控制 alen 后, 长问题的首token NLL 平均仍高 %+.4f; 内部 NLL 高 %+.4f"
              % (sum(gaps_f) / len(gaps_f), sum(gaps_r) / len(gaps_r)))

    # ---- 判决 ----
    print("\n[判决对照预登记规则]")
    if inc_r and inc_f:
        print("  ★命中 **R1**: M1 与 M2 都随问题长度上升 ⇒ 长度是难度 ⇒ **窗口方向彻底放下**")
    elif inc_f and not inc_r:
        print("  ★命中 **R2**: 只有 M2 上升 ⇒ 长度专门影响绑定 ⇒ 重训 W=32 值得试")
    else:
        print("  ○ 三条规则均未命中 ⇒ 分层本身不解释失败 ⇒ 回到 Δ 诊断(§41)")
    print("\n[eval 侧同口径, 仅作参考: eval 首token acc 0.1078 / 内部 acc 0.2560, 基线接近随机]")
    return 0


if __name__ == "__main__":
    sys.exit(main())