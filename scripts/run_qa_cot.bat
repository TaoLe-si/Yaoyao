@echo off
REM Overnight QA + chain-of-thought on APE calc + alpaca subset + digit CoT.
REM Budget: 8 shards x 400 steps = 3200 steps, 9.5M, no wiki/code mix.
setlocal EnableExtensions
cd /d D:\TaoVm

if not exist build mkdir build
if not exist data mkdir data

if not exist build\train_shards.exe (
  call scripts\build_train_shards.bat
  if errorlevel 1 goto :fail
)
if not exist build\h2r_cpu.exe (
  call scripts\build_h2r_cpu.bat
  if errorlevel 1 goto :fail
)
if not exist build\build_corpus.exe (
  call scripts\build_corpus_tools.bat
  if errorlevel 1 goto :fail
)
if not exist build\tok_qa.bbp (
  echo FAIL missing build\tok_qa.bbp
  goto :fail
)

echo [1/3] prep QA+CoT mix
if not exist data\qa_cot\train.jsonl (
  node scripts\prep_qa_cot.mjs
  if errorlevel 1 goto :fail
) else (
  echo skip prep, data\qa_cot already exists
)

echo [2/3] shard corpus
if not exist data\sft_qa_cot\shard_00000.bin (
  build\build_corpus.exe --out data\sft_qa_cot --tokenizer build\tok_qa.bbp --format jsonl --stage core --dialogue-out --docs-per-shard 5000 --min-bytes 32 data\qa_cot\train.jsonl
  if errorlevel 1 goto :fail
) else (
  echo skip corpus, data\sft_qa_cot already exists
)

if exist build\L1_qa_cot (
  echo FAIL output build\L1_qa_cot already exists - never overwrite
  goto :fail
)

echo [3/3] train
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
set TAO_SHUFFLE_SEED=20260912

build\train_shards.exe data\sft_qa_cot build\tok_qa.bbp build\L1_qa_cot 400 32 32
echo EXITCODE_QA_COT=%errorlevel%
if errorlevel 1 goto :fail
echo QA_COT_TRAIN_DONE
echo   set TAO_TOKENIZER=D:\TaoVm\build\tok_qa.bbp
echo   node scripts\eval_real.mjs build\L1_qa_cot\final.dsb data\qa_cot\heldout_qa.jsonl 200
echo   node scripts\eval_cot.mjs build\L1_qa_cot\final.dsb data\qa_cot\heldout_cot.jsonl
exit /b 0

:fail
echo EXITCODE_QA_COT=%errorlevel%
exit /b 1
