"""训练步的 padding 占比 —— 可能是**最大且完全等价**的一块浪费。

**假设 P-PAD**: 出货训练用 `encode_examples(rows, tok, 512)` 把每行**补齐到 512**,
但真实行长远短于 512。`train_dialogue` 里 `valid = pos < lengths_b` 把补齐位从 loss 里
mask 掉了, **可是网络仍然把补齐位算了一遍**。

★为什么"丢掉补齐位"是**精确等价**而不是近似:
  · 补齐位在序列**尾部**(positions ≥ length), 因果注意力 ⇒ 尾部的补齐位**不可能**影响
    它前面的任何位置;
  · loss 侧本来就用 `valid` 把它们 mask 掉了;
  · Engram 分支只看每个位置自己的 n-gram, 也不跨位置;
  ⇒ 把批的张量从 L 截到 `lengths_b.max()`, loss 与梯度**完全相同**(只有浮点归约形状
    变化带来的机器精度差异, 规则 14)。

**预登记**:
  - **P1**: 真实行长度的中位数远小于 512 ⇒ 截断能省下可观算力。
    **否证**: 中位数 ≥ 0.8·SEQ ⇒ P1 被否(截断基本没用)。
  - **P2**: 每步随机抽 batch 时, `lengths_b.max()` 通常也显著小于 SEQ。
    **否证**: 随机抽 64 个的最大长度 ≥ 0.95·SEQ ⇒ P2 被否(长行会拉满)。
  - **P3**: 截断后 loss/梯度在容差内一致。**否证**: 超出机器精度容差 ⇒ P3 被否。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
V11 = os.path.join(HERE, "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")
SEQ = 512
BS = 256
MICRO = 64


def main():
    from tokenizers import Tokenizer
    from dialogue_small import split_records, encode_examples, answer_loss_sums
    from model import RCfg, EngramRuleLM
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    tr_rows, _, _ = split_records(ck["extra"]["raw_dialogue_rows"], train_fraction=0.8,
                                  seed=29, return_meta=True)
    x, starts, lengths = encode_examples(tr_rows, tok, SEQ)
    L = x.shape[1]
    lens = lengths.tolist()
    lens_sorted = sorted(lens)
    n = len(lens)
    med = lens_sorted[n // 2]
    p90 = lens_sorted[int(0.9 * n)]
    print("训练行数 %d | 张量长度 %d | 行长度 min=%d 中位=%d p90=%d max=%d"
          % (n, L, min(lens), med, p90, max(lens)))
    print("★中位/SEQ = %.3f ⇒ P1 %s" % (med / L, "成立" if med < 0.8 * L else "被否"))
    print("  全量平均填充率 = %.1f%% (即 %.1f%% 的位置是补齐位, 网络仍然全算)"
          % (100 * (1 - sum(lens) / (n * L)), 100 * (1 - sum(lens) / (n * L))))

    # ---- P2: 随机抽 batch 时 max 长度有多长 ----
    g = torch.Generator().manual_seed(7)
    t = torch.tensor(lens)
    maxima = []
    for _ in range(200):
        idx = torch.randint(0, n, (MICRO,), generator=g)
        maxima.append(int(t[idx].max()))
    maxima.sort()
    print("  200 次随机抽 %d 行: batch 最大长度 中位=%d p90=%d" % (MICRO, maxima[100], maxima[180]))
    print("  ⇒ 期望截断到的 L 中位 %.0f (%.0f%%) ⇒ P2 %s"
          % (maxima[100], 100 * maxima[100] / L,
             "成立" if maxima[100] < 0.95 * L else "被否"))
    # 排序后再抽(batch 内长度齐)的上界
    print("  若先按长度排序再组批, 理论可把 L 降到中位 %d (%.0f%%)"
          % (med, 100 * med / L))

    # ---- P3: 截断等价性 ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    m = EngramRuleLM(RCfg(**ck["cfg"])).to(dev)
    m.engram.alive[:] = True
    sub = x[:8]
    lb = lengths[:8]
    Lmax = int(lb.max())

    def run(xx):
        xb = xx.to(dev)
        sb = starts[:8].to(dev)
        ll = lb.to(dev)
        pos = torch.arange(xb.shape[1], device=dev).view(1, -1)
        valid = pos < ll.view(-1, 1)
        logits = m(xb)[:, :-1, :]
        tok_loss, first_loss, _, _ = answer_loss_sums(logits, xb[:, 1:], valid, sb)
        # answer_loss_sums 返回的是**每行一个标量**(0 维张量)⇒ 不能 cat, 直接逐行比。
        return torch.stack([tok_loss.reshape(-1), first_loss.reshape(-1)])
    a = run(sub[:, :L])
    b = run(sub[:, :Lmax])
    rel = float((a - b).abs().max() / max(float(a.abs().max()), 1e-12))
    print("\n[P3] 截断到 max(lengths)=%d (原 %d) 后: 每行 loss 相对最大差 %.3e ⇒ %s"
          % (Lmax, L, rel, "通过(机器精度量级)" if rel < 1e-5 else "★被否"))
    print("      全长逐行 = %s" % ["%.6f" % v for v in a[0].tolist()])
    print("      截断逐行 = %s" % ["%.6f" % v for v in b[0].tolist()])

    # ★关键前提: 只有 bucket_step>0(固定步长)时, 窗口才与 L 无关, 截断才等价。
    #   bucket_step==0 是 legacy 动态步长, _bucket_step(L) 随 L 变 ⇒ 截断会改变
    #   **每一个位置**的窗口 ⇒ 不等价, 必须禁止。
    bs = ck["cfg"]["bucket_step"]
    print("\n      v11 bucket_step=%d ⇒ %s" % (bs, "截断安全" if bs > 0
                                              else "★legacy 动态步长, 截断会改变窗口"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
