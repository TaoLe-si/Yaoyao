"""§69.38 只读分解：自由生成差，到底是「重复」还是「别的原因」。

── 为什么先做这个 ──────────────────────────────────────────────────
§69.37 已确认「加训练量无效」。接下来要改**训练内容**（语料 / 损失 / 解码），
但改之前必须先知道**当前的主要失败模式是什么**，否则改了也不知道有没有打到点子上。
我此前一直把「重复」当成主因（repeat_rate 中位 0.22~0.28），但**没有验证过
「不重复的那批是不是就好了」**。这正是 AGENTS 规则 9 要的：判据必须先可失败。

── 三个可证伪读数 ──────────────────────────────────────────────────
| 编号 | 读什么 | 若成立意味着 |
|---|---|---|
| **M1** | 按 repeat_rate 分档后，低重复档的 F1 是否显著更高 | 若两档 F1 接近 ⇒ **重复不是主因**，我此前归因错了 |
| **M2** | generated_tokens 与参考答案长度的关系 | 若普遍偏短 ⇒ 模型「早早收尾」，不是「说错」而是「没说够」 |
| **M3** | eos 命中率 | 若接近 100% ⇒ 模型知道何时停，问题在停之前说什么 |

用法：`python probe_failure_modes.py`（只读已落盘 JSON，不生成、不训练）
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fn):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def f1m(rs):
    return sum(r["char_f1"] for r in rs) / len(rs) if rs else float("nan")


def main():
    print("=" * 78)
    print("失败模式分解（只读已落盘的抽样 200）")
    print("=" * 78)

    for tag, fn in (("满步 wd=0", "qa_held_s200.jsonl"),
                    ("满步+500步", "qa_held_s200_e500.jsonl")):
        rows = load(fn)
        if rows is None:
            print("\n[%s] %s 不存在，跳过" % (tag, fn))
            continue
        n = len(rows)
        print()
        print("【%s】%s  n=%d" % (tag, fn, n))

        # ── M1 按重复率分档 ────────────────────────────────────────
        buckets = {"rep<0.1": [], "0.1<=rep<0.3": [], "0.3<=rep<0.5": [], "rep>=0.5": []}
        for r in rows:
            v = r["repeated_token_rate"]
            k = ("rep<0.1" if v < 0.1 else "0.1<=rep<0.3" if v < 0.3
                 else "0.3<=rep<0.5" if v < 0.5 else "rep>=0.5")
            buckets[k].append(r)
        print("  M1 按 repeat_rate 分档：")
        for k, rs in buckets.items():
            print("    %-14s n=%-4d (%2.0f%%)  F1 mean %.4f"
                  % (k, len(rs), 100.0 * len(rs) / n, f1m(rs)))
        lo = [r for r in rows if r["repeated_token_rate"] < 0.5]
        hi = [r for r in rows if r["repeated_token_rate"] >= 0.5]
        d = f1m(hi) - f1m(lo)
        print("    ⇒ rep>=0.5 的 F1 比 rep<0.5 %s %.4f（%.1f%%）"
              % ("低" if d < 0 else "高", abs(d), 100 * abs(d) / max(f1m(lo), 1e-9)))
        if abs(d) < 0.05:
            print("    ★两档 F1 接近 ⇒ **重复不是主因**，此前归因需修正")
        else:
            print("    ⇒ 重复确实拉低 F1，但低重复档也只有 %.4f ⇒ 不是唯一原因"
                  % f1m(lo))

        # ── M2 长度 ───────────────────────────────────────────────
        gt = [r["generated_tokens"] for r in rows]
        print("  M2 生成长度：med=%d mean=%.1f min=%d max=%d"
              % (st.median(gt), sum(gt) / n, min(gt), max(gt)))
        short = [r for r in rows if r["generated_tokens"] < 20]
        print("    <20 token 的有 %d/%d（%.0f%%），这批 F1 mean %.4f"
              % (len(short), n, 100.0 * len(short) / n, f1m(short)))

        # ── M3 eos ────────────────────────────────────────────────
        ne = sum(1 for r in rows if r["eos"])
        print("  M3 eos=true: %d/%d（%.0f%%）" % (ne, n, 100.0 * ne / n))
        if ne / n > 0.9:
            print("    ⇒ 模型**知道何时停** ⇒ 问题在「停之前说什么」，不是「不会收尾」")

    print()
    print("=" * 78)
    print("【判据的可失败性】")
    print("=" * 78)
    print("  M1 若两档 F1 接近 ⇒ 「重复是主因」这个判断**被否**。")
    print("  这是本节唯一能推翻我此前归因的读数，所以先做它，再决定改什么。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
