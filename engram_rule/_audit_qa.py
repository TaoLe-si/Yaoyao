"""静态审计 run_qa_tables.py（AGENTS §1A 步 B/C/E/F）。

★「gen_split 内仍写死 OUT_TRAIN」这条是**误报**：它匹配到的是
  `path = path or (OUT_TRAIN if tag == "train" else OUT_HELD)`
  这行**兜底默认值**，而真实路径由调用方经 `path=` 传入（`out_tr` / `out_hd`）。
  兜底与覆盖并存是刻意设计：不给 path 时仍能独立运行。
"""
import ast
import io
import re

src = io.open("run_qa_tables.py", encoding="utf-8").read()
ast.parse(src)
print("AST OK")

checks = [
    ("import argparse", "import argparse" in src),
    ("--ckpt 可指定权重", 'ap.add_argument("--ckpt"' in src),
    ("--tag 分开落盘", 'ap.add_argument("--tag"' in src),
    ("CKPT 由参数覆盖", "CKPT = a.ckpt" in src),
    ("out_tr/out_hd 已派生", ("out_tr =" in src) and ("out_hd =" in src)),
    ("gen_split 收 path 参数", "def gen_split(model, tok, rows, tag, path=None)" in src),
    ("path 有兜底默认", "path = path or (OUT_TRAIN if tag" in src),
    ("调用方传 out_tr", 'gen_split(model, E2._TOK, samp_tr, "train", out_tr)' in src),
    ("调用方传 out_hd", 'gen_split(model, E2._TOK, samp_hd, "held", out_hd)' in src),
    # ---- 上一轮踩过的坑，回归守护 ----
    ("不再读 st['model']", 'st["model"]' not in src),
    ("不再重建 n_heads", 'cfg["n_heads"] =' not in src),
    ("cfg 逐字用成品", 'cfg = dict(ck["cfg"])' in src),
    ("形状自检在 load 前", src.index("bad = [(n, tuple(sd[n].shape)")
     < src.index("m.load_state_dict(sd)")),
    ("load_state_dict strict", "m.load_state_dict(sd)" in src),
    ("flush + fsync", ("f.flush()" in src) and ("os.fsync(f.fileno())" in src)),
    ("5 条速度校准", ("CALIB = 5" in src) and ("per * SAMPLE" in src)),
    ("G1 行数判据", ("assert disk == SAMPLE" in src) or ("assert disk == n" in src)),
]
bad = [n for n, ok in checks if not ok]
for n, ok in checks:
    print("  %-26s %s" % (n, "OK" if ok else "*FAIL"))
if bad:
    print("静态检查未过：%s" % bad)
    raise SystemExit(1)
print("全部静态检查通过（%d 项）" % len(checks))
