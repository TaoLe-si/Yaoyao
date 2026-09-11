@echo off
cd /d D:\TaoVm
del build\PDF_DONE.txt build\pdf_extract.log 2>nul
echo EXTRACT_START %TIME% > build\pdf_extract.log
python -m pip install --quiet pypdf >> build\pdf_extract.log 2>&1
echo pip_exit=%errorlevel% >> build\pdf_extract.log
python scripts\extract_pdf.py >> build\pdf_extract.log 2>&1
echo py_exit=%errorlevel% >> build\pdf_extract.log
echo EXTRACT_DONE %TIME% >> build\pdf_extract.log
echo done > build\PDF_DONE.txt
