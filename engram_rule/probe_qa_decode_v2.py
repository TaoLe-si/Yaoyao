"""★对 v2 的大权重做**完整问答解码**，看自由生成质量（零抽样）。

**为什么需要这个脚本**：`evaluate_generations` 是出货解码路径，但 §67 的格点只落了
`probe_nll`（教师强制口径）。NLL 低不等于自由生成好 —— §56 已证两者可以脱节。
这个脚本专门看**自由生成**的 EM / char-F1 / EOS 率 / lcp 分布。

**口径**（与 `probe_train_qa_report_st400.py` 完全一致，可比）：
- greedy、`max_new_tokens=400`（§60.5：语料 gold_len 最大 366，设 90 会白丢样本）
- `context_limit=512`、utf8-safe、批生成
- **只测 train 侧**（桃的指示：不再跑 eval）
- **零抽样**：探针集全部 31,147 行

**不重训、不改模型**，只加载 checkpoint 解码。
"""
import json
import os
import sys
from collections import Counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
MAX_NEW = 400


def load_dialogue_small():
    """按路径加载 `dist/dialogue_small.py`（主目录那份 import 不了 train）。"""
    import importlib.util
    path = os.path.join(HERE, "dist", "dialogue_small.py")
    spec = importlib.util.spec_from_file_location("ds_qa", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ds_qa"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "requests"):
        import requests
        mod.requests = requests
    return mod


DS = load_dialogue_small()


def lcp_len(a, b):
    """公共前缀长度（字符级）。"""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def eval_one(ckpt_path, rows, tok, tag):
    import model as M
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.manual_seed(int(ck["cell"]["seed"]))
    model = M.EngramRuleLM(M.RCfg(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    m, outs = DS.evaluate_generations(
        model, tok, rows, max_new_tokens=MAX_NEW, device=DEV,
        context_limit=SEQ, temperature=0.0, batch_min_size=8)

    em = f1 = 0.0
    lcps, eos_ok = [], 0
    for o in outs:
        gen, ref = o["generated"], o["reference"]
        em += float(DS.normalize_text(gen) == DS.normalize_text(ref))
        f1 += DS.char_f1(gen, ref)
        lcps.append(lcp_len(gen, ref))
        if gen.rstrip().endswith(("。", "！", "？", "”", ")", "）")) or "[EOS]" in gen:
            eos_ok += 1
    n = len(outs)
    rec = dict(tag=tag, ckpt=os.path.basename(ckpt_path), n=n,
               params=int(ck["cell"].get("params", 0)) or None,
               em=em / n, char_f1=f1 / n,
               lcp_mean=sum(lcps) / n,
               lcp_zero=sum(1 for v in lcps if v == 0) / n,
               eos_rate=eos_ok / n,
               raw=m)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return rec, outs


def main():
    from tokenizers import Tokenizer
    import exp_scaling_grid_v2 as e

    rows = e.load_rows()
    _tr, probe_rows = e.split_rows(rows)
    e.assert_no_leak(_tr, probe_rows)

    # v11 的 tokenizer 与本轮实验同源（cfg 取自 v11，只改 d / n_slots）
    v11 = torch.load(os.path.join(HERE, "archive",
                                  "engram_rule_dialogue_bpe_answer_disjoint_v11.pt"),
                     map_location="cpu", weights_only=False)
    tok = Tokenizer.from_str(v11["tokenizer"]["json"])

    targets = [
        ("engram_rule_scaling_v2_n124593_d32_wd0p0.pt", "N=124593 d=32  (探针 NLL 2.9464)"),
        ("engram_rule_scaling_v2_n124593_d64_wd0p0.pt", "N=124593 d=64  (探针 NLL 2.5646)"),
        ("engram_rule_scaling_v2_n31148_d128_wd0p0.pt", "N=31148  d=128 (探针 NLL 3.0094)"),
        ("engram_rule_scaling_v2_n31148_d64_wd0p0.pt", "N=31148  d=64  (探针 NLL 2.9555)"),
    ]
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print("解码 %d / %d 行（train 侧，零抽样）| device=%s | max_new=%d"
          % (min(n_rows, len(probe_rows)), len(probe_rows), DEV, MAX_NEW), flush=True)

    out_jsonl = os.path.join(HERE, "qa_decode_v2.jsonl")
    summary = []
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for name, tag in targets:
            p = os.path.join(HERE, name)
            if not os.path.exists(p):
                print("  跳过（不存在）%s" % name, flush=True)
                continue
            rec, outs = eval_one(p, probe_rows[:n_rows], tok, tag)
            summary.append(rec)
            for o in outs:
                f.write(json.dumps(dict(tag=tag, **o), ensure_ascii=False) + "\n")
            print("  %-34s EM=%.4f  char-F1=%.4f  lcp均值=%.1f  lcp=0占比=%.4f  EOS率=%.4f"
                  % (tag, rec["em"], rec["char_f1"], rec["lcp_mean"],
                     rec["lcp_zero"], rec["eos_rate"]), flush=True)
    with open(os.path.join(HERE, "qa_decode_v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("\n已写出 %s 与 qa_decode_v2_summary.json" % os.path.basename(out_jsonl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
