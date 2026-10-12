"""只读探针：读出货出 checkpoint 的真实 cfg，推出「注意力最远能看多远」。

不改任何实现文件，不训练，只读一个 .pt 的 cfg 字典并做算术。
用途：`train_cot_smoke.py` 的 T4 需要这些真实数字做断言，不能拍脑袋写死。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

V11 = os.path.join(HERE, "archive", "engram_rule_dialogue_bpe_answer_disjoint_v11.pt")

KEYS = ["d", "n_heads", "head_dim", "n_layers", "ngram_max", "n_hash", "n_slots",
        "min_count", "n_buckets", "k_buckets", "bucket_step", "window_max",
        "pos_abs_max", "near_first", "rope_dim", "gate_rank", "ffn_mult",
        "use_pos_abs", "use_pos_bucket", "clamp_window", "local_norm"]


def main():
    ck = torch.load(V11, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    print("=" * 78)
    print("出货 checkpoint 的真实 cfg（只读）")
    print("=" * 78)
    for k in KEYS:
        if k in cfg:
            print("  %-16s = %r" % (k, cfg[k]))

    nb = int(cfg.get("n_buckets", 64))
    bs = int(cfg.get("bucket_step", 0))
    W = int(cfg.get("window_max", 16))
    pa = int(cfg.get("pos_abs_max", 512))

    print()
    print("=" * 78)
    print("「注意力最远能看多远」——纯算术，不需要跑")
    print("=" * 78)
    print("  bucket_step            = %d  (0 = legacy 动态步长)" % bs)
    if bs > 0:
        reach = nb * bs
        print("  固定步长 ⇒ 最远回看   = n_buckets × bucket_step = %d × %d = %d token"
              % (nb, bs, reach))
        print("  窗口宽度 window_max    = %d token（连续区间，不是稀疏采样）" % W)
        print("  ⇒ 覆盖的**最远那个**位置 = %d - %d + 1 = %d"
              % (reach, W - 1, reach - W + 1))
    else:
        step = -(-512 // nb)          # legacy: 按 L 动态，这里先按 L=512 估
        print("  legacy 动态步长 ⇒ step = ceil(L/%d)，随 L 变" % nb)
        print("  按 L=512 估算：step = %d，最远回看 = %d token" % (step, nb * step))
    print("  pos_abs_max            = %d  （位置表与桶偏置表的长度）" % pa)
    print()
    print("  ★ 关键：即使 pos_abs_max 放大到 1,000,000，**可达距离仍由上式决定**")
    print("    ⇒ 位置表变大 ≠ 模型能看到更远。这是两条独立的通路。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
