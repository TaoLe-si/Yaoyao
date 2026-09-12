@echo off
REM 基础问答：只用 alpaca-zh 对话，9.5M 默认形状，不混 wiki/code。
REM 慢训预算：约 10 片 × 400 步 ≈ 4000 步（不要用 5000 步/片）。
setlocal EnableExtensions
cd /d D:\TaoVm

if not exist build mkdir build
if not exist data mkdir data

echo [1/5] build tools
call scripts\build_bpe_train.bat
if errorlevel 1 goto :fail
call scripts\build_corpus_tools.bat
if errorlevel 1 goto :fail
call scripts\build_train_shards.bat
if errorlevel 1 goto :fail
call scripts\build_h2r_cpu.bat
if errorlevel 1 goto :fail

echo [2/5] prep alpaca QA
if not exist data\alpaca_qa\train.jsonl (
  node scripts\prep_alpaca_qa.mjs
  if errorlevel 1 goto :fail
) else (
  echo skip prep, data\alpaca_qa already exists
)

echo [3/5] train tokenizer on QA text
if not exist build\tok_qa.bbp (
  build\bpe_train.exe build\tok_qa.bbp 16123 90 data\alpaca_qa\tok_text.txt
  if errorlevel 1 goto :fail
) else (
  echo skip tokenizer, build\tok_qa.bbp already exists
)

echo [4/5] shard dialogue corpus
if not exist data\sft_alpaca_zh\shard_00000.bin (
  build\build_corpus.exe --out data\sft_alpaca_zh --tokenizer build\tok_qa.bbp --format jsonl --stage core --dialogue-out --docs-per-shard 5000 --min-bytes 32 data\alpaca_qa\train.jsonl
  if errorlevel 1 goto :fail
) else (
  echo skip corpus, data\sft_alpaca_zh already exists
)

if exist build\L1_sft_small (
  echo FAIL output build\L1_sft_small already exists — never overwrite
  goto :fail
)

echo [5/5] train QA model
set TAO_ALLOW_TOKENIZER=1
set TAO_TOKENIZER=D:\TaoVm\build\tok_qa.bbp
set TAO_CPU_THREADS=8
set TAO_CFG_LAYERS=2
set TAO_CFG_D=512
set TAO_CFG_S=128
set TAO_CFG_M=512
set TAO_CFG_E=1024
set TAO_CFG_DK=64
set TAO_CFG_VOCAB=16384
set TAO_OPT_OFFLOAD=0
set TAO_OPT_STATE=1
set TAO_LR=0.001
set TAO_LR_DECAY_START=400
set TAO_LR_DECAY_STEPS=3200
set TAO_LR_MIN=0.00001
set TAO_GRAD_CLIP=1.0
set TAO_SHUFFLE_SEED=20251120

build\train_shards.exe data\sft_alpaca_zh build\tok_qa.bbp build\L1_sft_small 400 32 32
echo EXITCODE_RUN_SFT=%errorlevel%
if errorlevel 1 goto :fail
echo QA_TRAIN_DONE eval with:
echo   set TAO_TOKENIZER=D:\TaoVm\build\tok_qa.bbp
echo   node scripts\eval_real.mjs build\L1_sft_small\final.dsb data\alpaca_qa\heldout.jsonl 200
exit /b 0

:fail
echo EXITCODE_RUN_SFT=%errorlevel%
exit /b 1
