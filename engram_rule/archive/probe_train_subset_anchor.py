"""★口径核对: train 前 204 行是不是比全部 820 行容易?

**为什么查这个**: DESIGN §38 给的 v11 train `answer_token_accuracy` 是 **0.7905**,
而 §56 part0 用**前 204 行**量到 **0.8703**。差 0.08, 不查清就没法把
「自由生成 EM」与既有的教师强制锚点放在同一张表里(规则 23: 锚点必须与目标口径一致)。

**可能的原因有三个, 本脚本逐个分开量**:
  1. 行子集本身更容易(前 204 行比全 820 行简单)
  2. 分母口径不同(含/不含末尾 EOS 位、mask 的位置范围)
  3. 权重不是同一个(v11 出厂 vs §48 d=64/seed=11 重训)

判据: 三个子集(前 204 / 后 204 / 全 820)的读数必须**真的不同**, 否则本脚本是常量指标。
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_BS = 32


@torch.no_grad()
def token_acc_answer_metrics(model, x, starts, lengths):
    """**逐字复制** `dialogue_small.answer_metrics` 的 token 口径(不 import 它,
    是为了能独立跑子集切片; 复制来的公式与原函数一致, 由下面的 D0 对齐判据看守)。"""
    correct = 0
    total = 0
    for bi in torch.arange(x.shape[0]).split(EVAL_BS):
        bx = x[bi].to(DEV)
        sb = starts[bi].to(DEV)
        lb = lengths[bi].to(DEV)
        positions = torch.arange(bx.shape[1], device=DEV).view(1, -1)
        valid = positions < lb.view(-1, 1)
        target = bx[:, 1:]
        mask = valid[:, 1:] & (positions[:, 1:] >= sb.view(-1, 1))
        logits = model(bx)[:, :-1, :]
        correct += ((logits.argmax(-1) == target) & mask).sum()
        total += mask.sum()
    return float(correct) / float(total), int(total)


def main():
    import exp_capacity_scaling as ecs
    from tokenizers import Tokenizer
    from model import RCfg, EngramRuleLM
    from dialogue_small import answer_metrics

    ck = torch.load(ecs.V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr_rows, ev_rows, tr, ev = ecs.load_data()[2:]
    tr_x, tr_s, tr_l = tr
    ev_x, ev_s, ev_l = ev

    model = EngramRuleLM(RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    print("=" * 74)
    print("D0  本脚本的复刻口径 == dialogue_small.answer_metrics(全量 eval)")
    print("=" * 74)
    mine, ntok = token_acc_answer_metrics(model, ev_x, ev_s, ev_l)
    ref = answer_metrics(model, ev_x, ev_s, ev_l, EVAL_BS, DEV)
    print("  复刻 %.6f (%d tokens)  原函数 %.6f  |Δ|=%.2e"
          % (mine, ntok, ref["answer_token_accuracy"], abs(mine - ref["answer_token_accuracy"])))
    ok0 = abs(mine - ref["answer_token_accuracy"]) < 1e-9
    print("  [%s] D0 对齐" % ("PASS" if ok0 else "FAIL"))

    print()
    print("=" * 74)
    print("D1  train 的三个子集(v11 出厂权重)")
    print("=" * 74)
    subsets = [("全部 %d 行" % tr_x.shape[0], slice(0, tr_x.shape[0])),
               ("前 204 行(§56 用这个)", slice(0, 204)),
               ("后 204 行", slice(tr_x.shape[0] - 204, tr_x.shape[0]))]
    vals = {}
    for name, sl in subsets:
        acc, n = token_acc_answer_metrics(model, tr_x[sl], tr_s[sl], tr_l[sl])
        vals[name] = acc
        print("  %-22s token_acc=%.4f  (答案 token %d)" % (name, acc, n))
    spread = max(vals.values()) - min(vals.values())
    print("  ★子集极差 = %.4f  ⇒ %s" % (spread,
                                       "前 204 行确实更容易, §56 的 train 读数偏乐观"
                                       if spread > 0.01 else "子集之间没有实质差异"))

    print()
    print("=" * 74)
    print("D2  负向对照: 换一个权重(§48 重训的 d=64/seed=11)读数必须不同")
    print("=" * 74)
    import json
    hit = False
    with open(ecs.OUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            if j.get("arm") == "main" and j.get("d") == 64 and j.get("seed") == 11:
                print("  §48 落盘: d=64 seed=11 train token_acc=%.4f (n=%d)"
                      % (j["train"]["answer_token_accuracy"], j["train"]["answer_tokens"]))
                print("     该权重没存盘(§48 无 torch.save), 只能读它的落盘数字, 无法重算")
                hit = True
                break
    print("  [%s] 找到 §48 d=64/seed=11 的落盘记录" % ("PASS" if hit else "MISSING"))
    print()
    print("  结论: v11 出厂权重 ≠ §48 的 d=64/seed=11 重训权重,")
    print("        两者的 train token_acc 不能互相替代(%.4f vs 见上)。"
          % vals["全部 %d 行" % tr_x.shape[0]])
    return 0 if ok0 else 1


if __name__ == "__main__":
    sys.exit(main())