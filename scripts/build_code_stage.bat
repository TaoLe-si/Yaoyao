@echo off
cd /d D:\TaoVm
set TAO_CORPUS_THREADS=8
echo CODE_START %TIME% > build\code_status.txt
build\build_corpus.exe --out D:\TaoVm\data\stage_code --tokenizer D:\TaoVm\build\tok_real_v1.bbp --format plain --doc-sep "@@DOC@@" --preserve-layout --stage interfere --min-bytes 64 --max-repeat 0.5 --docs-per-shard 20000 E:\taovm-data\code_docs.txt > build\code_build.log 2>&1
echo EXIT=%errorlevel% >> build\code_status.txt
echo CODE_DONE %TIME% >> build\code_status.txt
