"""★量 v2 的**端到端**耗时构成：从启动到落盘，每一步各花多少。

**为什么单独测这个**：训练步本身已优化过（§52 截断拿到 2×），但 N 从 738 涨到
124,593 后，**非训练阶段**的开销会跟着 N 线性甚至超线性增长：
  · `encode_examples(rows, tok, 512)` —— 逐行 BPE 编码，N 行就 N 次
  · `build_engram_from_examples(model, x, l)` —— **逐行** Python 循环 + 每行一次
    `slot_ids` 前向 + `index_add_`，N=124,593 就是 124,593 次迭代
  · `probe_and_pick_micro` —— 额外建一次模型 + 跑 1 步（**白跑**）
  · 探针评测 —— 31,147 行教师强制
  · checkpoint 落盘

**只测量，不改任何代码。** 每阶段单独计时并打印。
"""
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    import exp_scaling_grid_v2 as e
    from tokenizers import Tokenizer
    import model as M

    d = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 124593
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200   # 只跑 200 步采样，不外推全部

    T = {}
    t_all = time.time()

    t = time.time()
    rows = e.load_rows()
    tr, pr = e.split_rows(rows)
    e.assert_no_leak(tr, pr)
    T["1_读语料+切分+泄漏断言"] = time.time() - t
    print("1. 读语料 + 切分 + 泄漏断言           %7.2f s" % T["1_读语料+切分+泄漏断言"],
          flush=True)

    v11 = torch.load(os.path.join(HERE, "archive",
                                   "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                     map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(v11["tokenizer"]["json"])

    t = time.time()
    x, s, l = e.DS.encode_examples(tr[:n], tok, 512)
    T["2_训练集 BPE 编码"] = time.time() - t
    print("2. 训练集 BPE 编码（%d 行）          %7.2f s" % (n, T["2_训练集 BPE 编码"]),
          flush=True)

    t = time.time()
    px, ps, pl = e.DS.encode_examples(pr, tok, 512)
    T["3_探针集 BPE 编码"] = time.time() - t
    print("3. 探针集 BPE 编码（%d 行）          %7.2f s"
          % (len(pr), T["3_探针集 BPE 编码"]), flush=True)

    t = time.time()
    ck = dict(v11["cfg"]); ck["d"] = d; ck["n_heads"] = d // 16
    torch.manual_seed(11)
    model = M.EngramRuleLM(M.RCfg(**ck)).to(DEV)
    T["4_建模型+搬 GPU"] = time.time() - t
    print("4. 建模型 + 搬 GPU                    %7.2f s" % T["4_建模型+搬 GPU"], flush=True)

    t = time.time()
    e.DS.build_engram_from_examples(model, x, l)
    T["5_build_engram（逐行）"] = time.time() - t
    print("5. build_engram_from_examples        %7.2f s  ★逐行 %d 次"
          % (T["5_build_engram（逐行）"], n), flush=True)

    t = time.time()
    e.DS.build_engram_from_examples(model, x, l)
    T["5b_再跑一次（幂等性）"] = time.time() - t
    print("5b. 同样操作再来一次                 %7.2f s" % T["5b_再跑一次（幂等性）"],
          flush=True)

    # probe_and_pick_micro 内部走 make_model，后者依赖模块级 _TRAIN_X/_TRAIN_L
    e._TRAIN_X, e._TRAIN_S, e._TRAIN_L = x, s, l
    t = time.time()
    t = time.time()
    mb, probe_sec = e.probe_and_pick_micro(v11, x, s, l, d)
    T["6_probe_and_pick_micro"] = time.time() - t
    print("6. probe_and_pick_micro（白跑1步）   %7.2f s  (mb=%d)"
          % (T["6_probe_and_pick_micro"], mb), flush=True)

    torch.manual_seed(11)
    model = M.EngramRuleLM(M.RCfg(**ck)).to(DEV)
    e.DS.build_engram_from_examples(model, x, l)

    t = time.time()
    e.train_steps(model, x, s, l, steps, mb, 0.0)
    T["7_训练 %d 步" % steps] = time.time() - t
    print("7. 训练 %d 步                         %7.2f s  (%.3f s/步)"
          % (steps, T["7_训练 %d 步" % steps], T["7_训练 %d 步" % steps] / steps), flush=True)

    t = time.time()
    pm = e.probe_nll(model, px, ps, pl)
    T["8_探针评测（教师强制）"] = time.time() - t
    print("8. 探针评测（%d 行教师强制）         %7.2f s" % (len(pr), T["8_探针评测（教师强制）"]),
          flush=True)

    t = time.time()
    ckpt = os.path.join(HERE, "_timing_tmp.pt")
    torch.save(dict(cfg=model.c.__dict__, state_dict=model.state_dict()), ckpt)
    T["9_checkpoint 落盘"] = time.time() - t
    print("9. checkpoint 落盘                   %7.2f s  (%.1f MB)"
          % (T["9_checkpoint 落盘"], os.path.getsize(ckpt) / 1048576.0), flush=True)
    os.remove(ckpt)

    total = time.time() - t_all
    print()
    print("=" * 66)
    print("各阶段占比（端到端 %.1f s，其中训练只占 %d 步）" % (total, steps))
    print("=" * 66)
    for k in sorted(T, key=lambda z: -T[z]):
        print("  %-28s %7.2f s   %5.1f%%" % (k, T[k], 100.0 * T[k] / total))
    print("  %-28s %7.2f s   100.0%%" % ("合计", total))
    print()
    # 外推：训练 19468 步（N=124593, epochs=40）的端到端
    per_step = T["7_训练 %d 步" % steps] / steps
    fixed = total - T["7_训练 %d 步" % steps]
    full_steps = e.steps_for(n, epochs=40.0, floor=0)
    print("外推到完整 %d 步：训练 %.0f s + 固定开销 %.0f s = **%.0f s (%.1f h)**"
          % (full_steps, per_step * full_steps, fixed,
             per_step * full_steps + fixed, (per_step * full_steps + fixed) / 3600.0))
    with open(os.path.join(HERE, "e2e_timing_n%d_d%d.json" % (n, d)),
              "w", encoding="utf-8") as f:
        json.dump(dict(n=n, d=d, steps=steps, phases=T, total=total,
                       per_step=per_step, fixed=fixed,
                       full_steps=full_steps,
                       full_estimate=per_step * full_steps + fixed),
                  f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
