import sys
try:
    from pypdf import PdfReader
except ImportError:
    print("NO_PYPDF"); sys.exit(3)
src = r"D:\Backup\Downloads\DeepSeek_V41_Tech_Report.pdf"
r = PdfReader(src)
parts = []
for i, p in enumerate(r.pages):
    try:
        t = p.extract_text() or ""
    except Exception as e:
        t = "<<extract failed: %s>>" % e
    parts.append("\n===== PAGE %d =====\n" % (i + 1) + t)
open(r"D:\TaoVm\build\v41_report.txt", "w", encoding="utf-8").write("\n".join(parts))
txt = "\n".join(parts)
print("PAGES=%d CHARS=%d ENGRAM=%d" % (len(r.pages), len(txt), txt.count("Engram")))
