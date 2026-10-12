"""§69.39 收窄「全面崩」：看 F1 是否由**开头几个 token** 决定。

── 为什么做这个 ────────────────────────────────────────────────────
§69.38 分桶结论是「不是某类崩，是全面崩」，那是个笼统判断，没有可操作的点。
本节用**最长公共前缀（LCP）**把它收窄：若 F1 与「生成与参考答案的开头重合长度」
强相关 ⇒ 崩是**开头错**导致的连锁（错一步后面全偏），
而不是「整段随机」⇒ 这两者的对策完全不同。

── 判据（先登记）─────────────────────────────────────────────────
| 编号 | 预测 | 失败意味着 |
|---|---|---|
| **L1** | LCP=0 的行占相当比例（不是全部 0） | 若全 0 ⇒ 这个量无判别力，换尺子 |
| **L2** | LCP=0 组的 F1 **显著低于** LCP>0 组 | 若两组接近 ⇒ 开头对不对**不决定**整体质量，归因又要改 |
| **L3** | 报两组的 n | n<10 不当结论（规则 16 同源） |

★LCP 的口径：逐**字符**比（中文按字），不是按 token —— 因为落盘只有解码后的文本，
  重新 tokenize 不保证还原原 id 序列（§54 记过这个坑）。逐字符是**下界**，
  但足够回答「开头对不对」这个问题。

用法：`python probe_lcp.py`（只读已落盘 JSON，不生成不训练）
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def lcp_len(a, b):
    """最长公共前缀长度（逐字符）。空串 ⇒ 0。"""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def f1m(rs):
    return sum(r["char_f1"] for r in rs) / len(rs) if rs else float("nan")


def main():
    print("=" * 78)
    print("LCP 探针：F1 是否由开头几个字决定（只读已落盘抽样 200）")
    print("=" * 78)
    print("  口径：逐**字符**比 reference 与 generated 的最长公共前缀。")
    print("        （不按 token：落盘只有解码文本，重 tokenize 不保证还原原 id 序列）")

    for tag, fn in (("满步 wd=0", "qa_held_s200.jsonl"),
                    ("满步+500步", "qa_held_s200_e500.jsonl")):
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            print("\n[%s] %s 不存在，跳过" % (tag, fn))
            continue
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for r in rows:
            r["_lcp"] = lcp_len(r["reference"].strip(), r["generated"].strip())

        n = len(rows)
        lcps = [r["_lcp"] for r in rows]
        print()
        print("【%s】n=%d" % (tag, n))
        print("  LCP 分布：med=%d mean=%.2f  =0 的 %d 条（%.0f%%）  >=10 的 %d 条（%.0f%%）"
              % (st.median(lcps), sum(lcps) / n,
                 sum(1 for v in lcps if v == 0), 100.0 * sum(1 for v in lcps if v == 0) / n,
                 sum(1 for v in lcps if v >= 10), 100.0 * sum(1 for v in lcps if v >= 10) / n))

        zero = [r for r in rows if r["_lcp"] == 0]
        nz = [r for r in rows if r["_lcp"] > 0]
        print("  L2 分组：")
        print("    LCP=0   n=%-4d F1 mean %.4f" % (len(zero), f1m(zero)))
        print("    LCP>0   n=%-4d F1 mean %.4f" % (len(nz), f1m(nz)))
        d = f1m(zero) - f1m(nz)
        print("    差 = %+.4f（LCP=0 比 LCP>0 %s %.1f%%）"
              % (d, "低" if d < 0 else "高", 100 * abs(d) / max(f1m(nz), 1e-9)))

        # ★L1 的可失败性：两组都必须有足够样本，且差异要超出量级
        assert len(zero) > 0 and len(nz) > 0, "L1 失败：有一组是空的 ⇒ 这个量无判别力"
        if d < -0.05:
            print("    ⇒ **开头对不对显著决定整体质量** ⇒ 崩是「开头错 → 连锁偏」")
        elif d > 0.05:
            print("    ⇒ LCP=0 的反而更好 ⇒ 「开头对」不是关键，归因需再改")
        else:
            print("    ⇒ 两组接近 ⇒ 开头对不对**不决定**整体质量，归因需再改")

        # 细看 LCP>0 里 LCP 大的那批
        big = [r for r in rows if r["_lcp"] >= 10]
        if big:
            print("    LCP>=10 那批（n=%d）F1 mean %.4f  生成长度中位 %d"
                  % (len(big), f1m(big), st.median([r["generated_tokens"] for r in big])))
            print("    其中 EM=1 的 %d 条" % sum(r["em"] for r in big))

    print()
    print("=" * 78)
    print("【边界】")
    print("  · LCP 逐字符，是**下界**（token 级可能更高）。")
    print("  · 单 seed、分组后样本更小 ⇒ 无种子极差，只作方向指引（规则 16）。")
    print("  · 本节只回答「开头对不对是否决定整体」，不回答「怎么让开头对」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
