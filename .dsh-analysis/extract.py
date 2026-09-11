
import sys, json
from pypdf import PdfReader
p = r"D:\Backup\Downloads\DeepSeek_V41_Tech_Report.pdf"
r = PdfReader(p)
out = []
for i, pg in enumerate(r.pages):
    try:
        out.append(pg.extract_text() or "")
    except Exception as e:
        out.append("[ERR %s]" % e)
txt = "\n\n===== PAGE BREAK =====\n\n".join(out)
open(r"D:\TaoVm\.dsh-analysis\pdf_text.txt","w",encoding="utf-8").write(txt)
meta = r.metadata
print(json.dumps({"pages": len(r.pages), "chars": len(txt), "meta": {k:str(v) for k,v in (meta or {}).items()}}, ensure_ascii=False))
print(txt[:3000])
