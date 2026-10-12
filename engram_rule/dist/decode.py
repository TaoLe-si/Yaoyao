"""★离线解码器: 加载基准权重, 用**出货的** greedy_generate 生成回答。

**这就是"解码器"**: 不自写生成循环(§62.1 的教训), 全部走 `dialogue_small.py`
里那份已被 18 项等价性判据看守过的 `greedy_generate` / `greedy_generate_batch`。

用法:
    python decode.py                       # 交互式, 每行一个问题
    python decode.py "你最喜欢的食物是什么？"   # 单次
    python decode.py --file qs.txt          # 批量, 一行一问
    python decode.py --bench                # 只跑速度基准(204 行固定问题)

权重与 tokenizer 都从 checkpoint 里读, **不需要**原仓库的任何其他文件。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

DEFAULT_CKPT = "engram_rule_nsweep_n2952_d128_s462_wd0p0.pt"
MAX_NEW_TOKENS = 400          # §64 基准口径(语料 gold_len 最大 366)
CONTEXT_LIMIT = 512


def load(ckpt_path, device=None):
    """载入权重 + tokenizer + cfg。返回 (model, tok, cfg)。"""
    from tokenizers import Tokenizer
    from model import RCfg, EngramRuleLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(ck["tokenizer"]["json"])
    model = EngramRuleLM(RCfg(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, tok, ck["cfg"]


def answer(model, tok, question, device="cpu", history=None):
    """单问单答。走**出货的** greedy_generate。"""
    import dialogue_small as ds
    row = dict(question=question, answer="", history=history or [])
    g = ds.greedy_generate(model, tok, row, max_new_tokens=MAX_NEW_TOKENS,
                           device=device, context_limit=CONTEXT_LIMIT,
                           temperature=0.0)
    return g["generated"]


def answer_batch(model, tok, questions, device="cpu"):
    """批量(等长分组批生成, §51 的 3× 加速路径)。"""
    import dialogue_small as ds
    rows = [dict(question=q, answer="", history=[]) for q in questions]
    m, outs = ds.evaluate_generations(model, tok, rows,
                                      max_new_tokens=MAX_NEW_TOKENS,
                                      device=device, context_limit=CONTEXT_LIMIT,
                                      batch_min_size=8)
    return [o["generated"] for o in outs], m


def main():
    ap = argparse.ArgumentParser(description="Engram-Rule 离线解码器")
    ap.add_argument("question", nargs="*", default=None, help="要问的问题(可多个)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="权重文件路径")
    ap.add_argument("--device", default=None, help="cuda / cpu, 默认自动")
    ap.add_argument("--file", default=None, help="从文件读问题, 一行一个")
    ap.add_argument("--bench", action="store_true", help="跑速度基准")
    ap.add_argument("--info", action="store_true", help="只打印权重信息")
    a = ap.parse_args()

    path = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(HERE, a.ckpt)
    if not os.path.exists(path):
        raise SystemExit("找不到权重: %s" % path)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    model, tok, cfg = load(path, device)
    params = sum(p.numel() for p in model.parameters())
    print("权重: %s" % os.path.basename(path))
    print("  d=%d  n_layers=%d  n_heads=%d  参数量 %s  设备 %s  载入 %.1fs"
          % (cfg["d"], cfg["n_layers"], cfg["n_heads"],
             "{:,}".format(params), device, time.time() - t0))
    if a.info:
        print("  cfg = %s" % json.dumps(cfg, ensure_ascii=False))
        return 0

    if a.bench:
        qs = ["你最喜欢的食物是什么？"] * 64
        t0 = time.time()
        outs, m = answer_batch(model, tok, qs, device)
        el = time.time() - t0
        print("基准: 64 行批生成 %.1fs (%.3f s/行), EOS 率 %.2f"
              % (el, el / len(qs), m["eos_rate"]))
        print("  首行: %r" % outs[0][:40])
        return 0

    qs = list(a.question or [])
    if a.file:
        with open(a.file, "r", encoding="utf-8") as f:
            qs += [ln.strip() for ln in f if ln.strip()]
    if not qs:
        print("进入交互模式(空行退出)")
        try:
            while True:
                q = input("问> ").strip()
                if not q:
                    break
                t0 = time.time()
                print("答> %s   (%.2fs)" % (answer(model, tok, q, device),
                                            time.time() - t0))
        except (EOFError, KeyboardInterrupt):
            print()
        return 0

    if len(qs) == 1:
        t0 = time.time()
        print(answer(model, tok, qs[0], device))
        print("(%.2fs)" % (time.time() - t0), file=sys.stderr)
    else:
        outs, m = answer_batch(model, tok, qs, device)
        for q, o in zip(qs, outs):
            print("问> %s\n答> %s\n" % (q, o))
        print("(%d 问, EOS 率 %.2f)" % (len(qs), m["eos_rate"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
