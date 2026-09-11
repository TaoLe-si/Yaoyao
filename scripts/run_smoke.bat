@echo off
cd /d D:\TaoVm
build\build_corpus.exe --out build\corpus_smoke --tokenizer build\formal_tokenizer.bbp --format dialogue --stage core --docs-per-shard 5000 data\noffn_l6\conversations.txt > build\smoke.log 2>&1
echo SMOKE_EXITCODE=%errorlevel% >> build\smoke.log
