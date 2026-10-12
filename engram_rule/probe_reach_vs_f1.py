"""§69.40-D 同源配对：**可见 vs 不可见**两组的 F1 / LCP 差。

── 前提（已满足）───────────────────────────────────────────────────
`probe_question_reach.py` 已改成与 `run_qa_tables.py` **同源**：
同一抽样种子 12345、同一 n=200、同一来源（held）。所以两边的 row 索引可逐行配对。

── 判据（可失败，先登记）───────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **D1** | 不可见组的 F1 显著低于可见组 | 若接近 ⇒ 可达性**不是**「第一个字错」的原因 |
| **D2** | 不可见组的 LCP=0 比例高于可见组 | 若接近 ⇒ 差异不出在第一步，与 §69.39 冲突 |
| **D3** | 两组 n 都报 | 小样本不当结论（规则 16） |

用法：`python probe_reach_vs_f1.py`（只读两份 JSON，不生成不训练）
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def lcp_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def f1m(rs):
    """rs 里的记录用的是短键 `f1`（配对时改过名），不是原始 JSON 的 `char_f1`。"""
    return sum(r["f1"] for r in rs) / len(rs) if rs else float("nan")


def main():
    print("=" * 78)
    print("D：问题末尾可见性 vs 生成质量（同源逐行配对）")
    print("=" * 78)

    rp = os.path.join(HERE, "question_reach.json")
    qp = os.path.join(HERE, "qa_held_s200.jsonl")
    for p, nm in ((rp, "question_reach.json"), (qp, "qa_held_s200.jsonl")):
        if not os.path.exists(p):
            print("缺 %s，先跑对应脚本" % nm)
            return 1

    reach = json.load(open(rp, encoding="utf-8"))
    with open(qp, encoding="utf-8") as f:
        qa = [json.loads(l) for l in f if l.strip()]

    det = reach["detail"]
    print("  question_reach: ckpt=%s  sample=%s"
          % (reach["ckpt"], json.dumps(reach.get("sample"), ensure_ascii=False)))
    print("  qa_held_s200: 行数=%d" % len(qa))

    # ★口径自检：两边必须同源同长，否则配对无效
    if len(det) != len(qa):
        print("  ★行数不同（%d vs %d）⇒ **不同源，不做配对**" % (len(det), len(qa)))
        return 1
    if reach.get("sample", {}).get("seed") != 12345:
        print("  ★question_reach 的抽样种子不是 12345 ⇒ 不同源，不做配对")
        return 1
    print("  ⇒ 同源自检通过（seed=12345, n=%d, 来源=held）" % len(det))

    # ── 逐行配对 ────────────────────────────────────────────────────
    vis, unv = [], []
    for d, r in zip(det, qa):
        lcp = lcp_len(r["reference"].strip(), r["generated"].strip())
        rec = dict(f1=r["char_f1"], em=r["em"], lcp=lcp,
                   gtok=r["generated_tokens"], q=r["question"][:24])
        (vis if d["covered"] else unv).append(rec)

    n_v, n_u = len(vis), len(unv)
    print()
    print("【配对结果】")
    print("  可见（窗口覆盖问题末尾） n=%-4d  F1 mean %.4f  LCP=0 的 %d（%.1f%%）"
          % (n_v, f1m(vis), sum(1 for r in vis if r["lcp"] == 0),
             100.0 * sum(1 for r in vis if r["lcp"] == 0) / max(n_v, 1)))
    print("  不可见                   n=%-4d  F1 mean %.4f  LCP=0 的 %d（%.1f%%）"
          % (n_u, f1m(unv), sum(1 for r in unv if r["lcp"] == 0),
             100.0 * sum(1 for r in unv if r["lcp"] == 0) / max(n_u, 1)))
    d1 = f1m(unv) - f1m(vis)
    print("  D1: F1 差 = %+.4f（不可见比可见 %s %.1f%%）"
          % (d1, "低" if d1 < 0 else "高", 100 * abs(d1) / max(f1m(vis), 1e-9)))
    r_v = sum(1 for r in vis if r["lcp"] == 0) / max(n_v, 1)
    r_u = sum(1 for r in unv if r["lcp"] == 0) / max(n_u, 1)
    print("  D2: LCP=0 比例 差 = %+.4f（不可见比可见 %s %.1f 个百分点）"
          % (r_u - r_v, "高" if r_u > r_v else "低", 100 * abs(r_u - r_v)))

    # ── D3 小样本标注 ──────────────────────────────────────────────
    for tag, rs in (("可见", vis), ("不可见", unv)):
        if len(rs) < 10:
            print("  ★D3: %s 组 n=%d < 10 ⇒ 其均值**不当结论**" % (tag, len(rs)))

    # ── 判决 ────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("【判决】")
    print("=" * 78)
    if d1 < -0.05:
        print("  ⇒ **不可见组明显更差** ⇒ 可达性是「第一个字错」的一个真实成因。")
        print("     那 %.0f%% 不可见的样本是真缺口 ⇒ 支持开 `use_qpool`（选项 A）。"
              % (100.0 * n_u / len(det)))
    elif d1 > 0.05:
        print("  ⇒ 不可见组反而更好 ⇒ 可达性不是原因，且方向相反，归因需再改。")
    else:
        print("  ⇒ 两组 F1 接近（差 %.4f）⇒ **可达性不是「第一个字错」的原因**，" % d1)
        print("     「问题不可达 ⇒ 开 q_pool」这条**再次被否**。得另找。")

    # 抽 3 条不可见的给人看
    print()
    print("  不可见组样本（前 3 条，人工可核）：")
    for r in unv[:3]:
        print("    LCP=%-3d F1=%.3f  gtok=%-4d  q=%s…"
              % (r["lcp"], r["f1"], r["gtok"], r["q"]))
    print()
    print("  ★单 seed、200 条、分组后样本更小 ⇒ 无种子极差，只作方向指引（规则 16）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
