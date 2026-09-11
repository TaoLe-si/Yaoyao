@echo off
cd /d D:\TaoVm
set TAO_CORPUS_THREADS=8
echo WIKI_START %TIME% > build\wiki_status.txt
build\build_corpus.exe --out D:\TaoVm\data\stage_wiki --tokenizer D:\TaoVm\build\tok_real_v1.bbp --format plain --doc-sep "@@DOC@@" --preserve-layout --stage mixed --min-bytes 64 --max-repeat 0.5 --docs-per-shard 20000 E:\taovm-data\wiki_docs.txt > build\wiki_build.log 2>&1
echo EXIT=%errorlevel% >> build\wiki_status.txt
echo WIKI_DONE %TIME% >> build\wiki_status.txt
