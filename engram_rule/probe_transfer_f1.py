"""§69.47 可迁移样本专项：在那 732 条「问题骨架在 train 出现过」的 held 上看 F1。

## 为什么测这个

§69.46 实测：held 问题骨架命中 train 的只有 **2.35%**（732/31147），
bigram Jaccard **0.3060**。⇒ 有可迁移模式，但覆盖面窄。

「覆盖面窄」与「模式用不上」是两回事，必须分开：
- 若那 732 条 F1 **明显更高** ⇒ 模式在起作用，held EM 恒 0 只因覆盖 2.35%
  ⇒ 对策是**扩句式覆盖**
- 若那 732 条 F1 **和其余一样低** ⇒ 共享句式也没用 ⇒ 另有原因

## 判据（可失败）

| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **K1** | 命中组 F1 明显高于未命中组（差 > 0.05） | 若接近 ⇒ 模式用不上，覆盖不是瓶颈 |
| **K2** | 两组 n 都报 | n<10 不当结论（规则 16） |
| **K3** | 用**同一权重**（d=256 满步）+ 同一生成参数 | 不同 ⇒ 比的是两个东西 |

用法：`python probe_transfer_f1.py`（生成 732 条，约 3.5 分钟）
"""
import io
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp_scaling_grid_v2 as E2

CKPT = os.path.join(HERE, "engram_rule_scaling_v2_n124593_d256_wd0p0.pt")
OUT = os.path.join(HERE, "transfer_f1.json")

SLOTS = [
    (re.compile(r"\d+(?:\.\d+)?"), "§"),
    (re.compile(r"《[^》]*》"), "§"),
    (re.compile(r"[A-Za-z][A-Za-z0-9.\-]*"), "§"),
    (re.compile(r"[一-鿿]{2,4}(?:公司|集团|品牌|大学|学院|银行|医院)"), "§"),
]


def skel(t):
    s = t
    for p, r in SLOTS:
        s = p.sub(r, s)
    return s


def main():
    print("=" * 78)
    print("§69.47 可迁移样本专项（骨架命中的 held vs 未命中）")
    print("=" * 78)
    import model as M                      # ★规则 7
    E2.load_all()
    rows = E2.load_rows()
    tr, hd = E2.split_rows(rows)
    print("  train %d / held %d" % (len(tr), len(hd)))

    sk_tr = {skel(r["question"]) for r in tr}
    hit = [r for r in hd if skel(r["question"]) in sk_tr]
    unhit = [r for r in hd if skel(r["question"]) not in sk_tr]
    print("  骨架命中 train 的 held：%d 条（%.2f%%）"
          % (len(hit), 100.0 * len(hit) / len(hd)))
    print("  未命中：%d 条" % len(unhit))

    # K3：同一权重、同一生成参数
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    torch.manual_seed(E2.SEED)
    m = M.EngramRuleLM(M.RCfg(**cfg)).to(E2.DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    x, s, l = E2.encode_cached(tr, E2._TOK, E2.SEQ)
    info = E2.DS.build_engram_from_examples(m, x, l)
    assert info["alive_slots"] > 0, "alive 全 False ⇒ 残壳"
    print("  权重 = %s（K3 同权重同参数）" % os.path.basename(CKPT))

    # 未命中组抽样同数量（与命中组等 n 才可比）
    import random
    rnd = random.Random(12345)
    k = min(len(hit), 700)
    hit_s = rnd.sample(hit, k)
    unhit_s = rnd.sample(unhit, k)
    print("  各取 %d 条对拍（未命中组随机抽样，seed=12345）" % k)

    def run(rs, tag):
        f1s, ems = [], []
        t0 = __import__("time").time()
        for i, r in enumerate(rs):
            out = E2.DS.greedy_generate(m, E2._TOK, r, max_new_tokens=140,
                                        device=E2.DEV, context_limit=512,
                                        temperature=0.0, utf8_safe=True)
            g, ref = out["generated"], out["reference"]
            f1s.append(E2.DS.char_f1(g, ref))
            ems.append(int(g.strip() == ref.strip()))
            if (i + 1) % 200 == 0:
                print("    [%s] %d/%d  F1=%.4f  %.0fs"
                      % (tag, i + 1, len(rs), sum(f1s) / len(f1s),
                         __import__("time").time() - t0), flush=True)
        return f1s, ems

    print()
    print("  生成中（命中组）…")
    f1_h, em_h = run(hit_s, "hit")
    print("  生成中（未命中组）…")
    f1_u, em_u = run(unhit_s, "unhit")

    import statistics as st
    print()
    print("=" * 78)
    print("【K1】命中组 vs 未命中组")
    print("=" * 78)
    print("  骨架命中   n=%-4d F1 mean %.4f median %.4f  EM=%d"
          % (len(f1_h), sum(f1_h) / len(f1_h), st.median(f1_h), sum(em_h)))
    print("  未命中     n=%-4d F1 mean %.4f median %.4f  EM=%d"
          % (len(f1_u), sum(f1_u) / len(f1_u), st.median(f1_u), sum(em_u)))
    d = sum(f1_h) / len(f1_h) - sum(f1_u) / len(f1_u)
    print("  K1 差 = %+.4f（命中组%s %.1f%%）"
          % (d, "高" if d > 0 else "低", 100 * abs(d) / max(sum(f1_u) / len(f1_u), 1e-9)))

    # K2 小样本标注
    for tag, rs in (("命中", f1_h), ("未命中", f1_u)):
        if len(rs) < 10:
            print("  ★K2: %s 组 n=%d < 10 ⇒ 不当结论" % (tag, len(rs)))

    print()
    if d > 0.05:
        print("  ⇒ **命中组明显更好** ⇒ 可迁移模式确实在起作用，")
        print("     held EM 恒 0 只因覆盖率只有 2.35% ⇒ 对策是**扩句式覆盖**。")
    elif d < -0.05:
        print("  ⇒ 未命中组反而更好 ⇒ 方向相反，归因需再改。")
    else:
        print("  ⇒ 两组接近（差 %+.4f）⇒ **共享句式也没用**，" % d)
        print("     可迁移模式存在但用不上 ⇒ 覆盖面不是瓶颈，另有原因。")

    facts = dict(ckpt=os.path.basename(CKPT), n_hit_total=len(hit), n_unhit=len(unhit),
                 hit_rate=round(len(hit) / len(hd), 5), k=k,
                 hit=dict(n=len(f1_h), f1=sum(f1_h) / len(f1_h),
                          f1_med=st.median(f1_h), em=sum(em_h)),
                 unhit=dict(n=len(f1_u), f1=sum(f1_u) / len(f1_u),
                            f1_med=st.median(f1_u), em=sum(em_u)),
                 delta=d)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print()
    print("  已写出 %s" % os.path.basename(OUT))
    print("  ★单 seed、各 %d 条、无种子极差（规则 16）。" % k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
