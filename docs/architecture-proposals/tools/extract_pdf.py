#!/usr/bin/env python3
"""extract_pdf.py — 把论文 PDF 抽取为带页分隔符的纯文本。

用法：
    python extract_pdf.py <input.pdf> <output.txt>

依赖：
    python -m pip install pypdf

陷阱（本项目实际踩过）：
    Python 文本模式默认把 \n 写成 CRLF。若按 "\n\n===== PAGE BREAK =====\n\n" 切分，
    整篇会被当成 1 页。本脚本显式使用 newline="\n"，或读回时用
    正则 /\r?\n\r?\n===== PAGE BREAK =====\r?\n\r?\n/ 切分。
"""
import sys

def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    try:
        from pypdf import PdfReader
    except ImportError:
        print("pypdf 未安装：python -m pip install pypdf", file=sys.stderr)
        return 2
    reader = PdfReader(src)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        pages.append("===== PAGE %d =====\n%s" % (i, page.extract_text() or ""))
    text = "\n\n===== PAGE BREAK =====\n\n".join(pages)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("pages=%d chars=%d -> %s" % (len(reader.pages), len(text), dst))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
