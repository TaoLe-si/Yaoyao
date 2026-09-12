@echo off
REM 基础问答 held-out：空回复 / 乱码 / 复读 / bigramF1
setlocal
cd /d D:\TaoVm
set TAO_TOKENIZER=D:\TaoVm\build\tok_qa.bbp
set TAO_CPU_THREADS=8
set DSB=%1
if "%DSB%"=="" set DSB=build\L1_qa_cot\final.dsb
if exist data\qa_cot\heldout_qa.jsonl (
  node scripts\eval_real.mjs %DSB% data\qa_cot\heldout_qa.jsonl 200
) else (
  node scripts\eval_real.mjs %DSB% data\alpaca_qa\heldout.jsonl 200
)
echo EXITCODE_EVAL_QA=%errorlevel%
