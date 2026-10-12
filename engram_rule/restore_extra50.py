"""把被覆盖掉的 50 步读数从旧日志里**恢复**成 JSON。

★为什么需要它：`extra50.json` 现在只剩 500 步那份 —— 500 步那次跑的时候
  脚本写的是同一个文件名，把 50 步的原始读数覆盖了。
  50 步是「余量 −1.48%（超出噪声门槛）」的**唯一原始证据**，
  500 步是「已收敛」的证据。两个都得留，否则 §69.37 的对比表没法复核。

  现在脚本已改成按步数分文件，但**已经丢的那次**只能从 `extra50.log` 回推 ——
  那份日志里 head/tail/delta/单步 σ 都在，足够重建。

用法：`python restore_extra50.py`（只读日志，写一个新 JSON，不动任何已有文件）
"""
import ast
import io
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "extra50.log")
OUT = os.path.join(HERE, "extra_50_lr3e-05.json")


def main():
    if not os.path.exists(LOG):
        print("缺 extra50.log，无法回推")
        return 1
    txt = io.open(LOG, encoding="utf-8", errors="replace").read()

    # 只取 50 步那一段（日志里 step 编号最大到 50 的就是它；500 步的日志已被覆盖，
    # 现存 extra50.log 正是 50 步那次）
    steps_line = re.search(r"step\s+(\d+)/(\d+)\s+loss", txt)
    n_steps = int(steps_line.group(2)) if steps_line else 50

    head = float(re.search(r"首\s*\d+\s*步均值\s*=\s*([\d.]+)", txt).group(1))
    tail = float(re.search(r"末\s*\d+\s*步均值\s*=\s*([\d.]+)", txt).group(1))
    delta = float(re.search(r"变化\s*=\s*([+-][\d.]+)", txt).group(1))
    lr = float(re.search(r"恒定 lr\s*=\s*([\d.eE+-]+)", txt).group(1))

    # 逐点 loss：日志只在 step 1/10/20/... 打点，抽得出这些
    pts = [(int(m.group(1)), float(m.group(2)))
           for m in re.finditer(r"step\s+(\d+)/\d+\s+loss\s*([\d.]+)", txt)]

    # ★门槛：日志里没有 σ（那次跑的是旧判据），用**同分布的 500 步 σ** 代替
    #   —— 同权重、同 lr、同数据、同批大小 ⇒ 单步抖动分布同源，这个代用在文档里要写明。
    sd500 = None
    p500 = os.path.join(HERE, "extra_500_lr3e-05.json")
    if os.path.exists(p500):
        d5 = json.load(open(p500, encoding="utf-8"))
        sd500 = d5.get("sd")
    if sd500 is None:
        # 兜底：用 500 步那份 trace 直接算
        d5 = json.load(open(os.path.join(HERE, "extra50.json"), encoding="utf-8"))
        sd500 = st.pstdev(d5["trace"])

    k = n_steps // 5
    se = sd500 / (k ** 0.5)
    thr = 2 * se
    verdict = "still_improving" if (delta < 0 and abs(delta) >= thr) else "converged"

    facts = dict(
        source="restored from extra50.log（原 extra50.json 被 500 步那次覆盖）",
        ckpt="engram_rule_scaling_v2_n124593_d128_wd0p0.pt",
        base_lr=0.003, const_lr=lr, wd=0.0, steps=n_steps,
        bs=256, micro=64, seed=11,
        head=head, tail=tail, delta=delta,
        sd=sd500, se=se, threshold=thr, verdict=verdict,
        logged_points=pts,
        note="σ 用同分布的 500 步那份代替（同权重/同 lr/同数据/同批大小）；"
             "门槛 2σ/√k 是修好后的判据，原 50 步那次跑的是旧判据（只比符号）。",
    )
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)

    print("=" * 74)
    print("恢复 50 步读数（从 extra50.log 回推）")
    print("=" * 74)
    print("  steps      = %d" % n_steps)
    print("  const_lr   = %g" % lr)
    print("  head/tail  = %.6f / %.6f" % (head, tail))
    print("  delta      = %+.6f (%+.3f%%)" % (delta, 100 * delta / head))
    print("  σ(500步)   = %.6f   门槛 2σ/√%d = %.6f" % (sd500, k, thr))
    print("  |Δ| %.6f vs 门槛 %.6f  ⇒  %s" % (abs(delta), thr, verdict))
    print("  日志打点   = %d 个" % len(pts))
    print("  已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
