"""★自包含推理包: 把 model.py + dialogue_small.py 的精简副本 + 权重 + tokenizer
   打成一个不依赖原仓库的 dist/ 目录。

**为什么要精简副本而不是直接拷 dialogue_small.py**: 原文件顶部
    import requests                      ← 只用于 sample_ranges(下载语料), 推理不需要
    from tokenizers import Tokenizer,... ← 需要, 保留
    import train as train_module         ← 只用于 load_dialogue_checkpoint
    from train import DEV, evaluate, ... ← 只用于 main / save_dialogue_checkpoint
  ⇒ 这 3 处 import 都会让**离线包**在没装 requests 或缺 train.py 的机器上直接崩。
  精简副本去掉这 3 行(以及只用它们的函数), 其余**逐字不动**。
"""
import os, shutil, json, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist")
os.makedirs(DIST, exist_ok=True)
CKPT = "engram_rule_dialogue_bpe_answer_disjoint_d128_s11_st400.pt"

# 1) model.py 逐字拷(它只依赖 torch)
shutil.copy(os.path.join(HERE, "model.py"), os.path.join(DIST, "model.py"))
print("model.py 逐字拷贝")

# 2) dialogue_small.py: 只改 import 段, 其余逐字
src = open(os.path.join(HERE, "dialogue_small.py"), encoding="utf-8").read()
lines = src.splitlines(keepends=True)
out, skipped = [], []
for ln in lines:
    s = ln.strip()
    if s == "import requests":
        skipped.append("import requests              # 只用于 sample_ranges(下载语料)")
        continue
    if s.startswith("import train as train_module") or s.startswith("from train import"):
        skipped.append(s + "   # 只用于 load/save_dialogue_checkpoint 与 main")
        continue
    out.append(ln)
open(os.path.join(DIST, "dialogue_small.py"), "w", encoding="utf-8").write("".join(out))
print("dialogue_small.py: 去掉 %d 行 import, 其余逐字" % len(skipped))
for s in skipped: print("   - " + s.strip())
