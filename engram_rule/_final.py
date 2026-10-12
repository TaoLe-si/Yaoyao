"""★最终验收: 从 zip 解出来, 在干净目录里跑通 —— 这才是"能离机运行"的真证据。"""
import zipfile, tempfile, subprocess, sys, os, shutil
tmp = tempfile.mkdtemp(prefix="zip_test_")
with zipfile.ZipFile("engram_rule_decoder_d128_st400.zip") as z:
    z.extractall(tmp)
print("解出:", sorted(os.listdir(tmp)))
r = subprocess.run([sys.executable, "decode.py", "今天天气如何？"],
                   capture_output=True, text=True, cwd=tmp, encoding="utf-8")
print("returncode:", r.returncode)
print(r.stdout.strip())
print("STDERR:", r.stderr.strip()[-200:] if r.stderr.strip() else "(空)")
shutil.rmtree(tmp, ignore_errors=True)
