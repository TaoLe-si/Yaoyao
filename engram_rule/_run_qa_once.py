import io, os, sys, traceback
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["run_qa_tables.py",
            "--ckpt", "engram_rule_scaling_v2_n2952_d256_wd0p0.pt",
            "--tag", "n2952d256"]
LOG = "qa_n2952d256.log"
buf = io.StringIO()
old = sys.stdout
sys.stdout = buf
code = 0
try:
    src = io.open("run_qa_tables.py", encoding="utf-8").read()
    exec(compile(src, "run_qa_tables.py", "exec"),
         {"__name__": "__main__",
          "__file__": os.path.abspath("run_qa_tables.py")})
except SystemExit:
    code = 0
except BaseException:
    traceback.print_exc(file=buf)
    code = 1
finally:
    sys.stdout = old
io.open(LOG, "w", encoding="utf-8").write(buf.getvalue())
print("CODE=%d -> %s" % (code, LOG))
