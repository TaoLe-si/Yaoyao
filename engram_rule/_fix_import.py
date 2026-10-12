"""补 import argparse（幂等），然后跑 _audit_qa.py 复检。"""
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "run_qa_tables.py")
src = io.open(p, encoding="utf-8").read()
if "import argparse" not in src:
    src = src.replace("import json\nimport os", "import argparse\nimport json\nimport os", 1)
    io.open(p, "w", encoding="utf-8").write(src)
    print("added import argparse")
else:
    print("import argparse already present")

r = subprocess.run([sys.executable, "_audit_qa.py"], cwd=HERE,
                   capture_output=True, text=True, encoding="utf-8")
print(r.stdout)
print(r.stderr[-800:] if r.stderr else "")
