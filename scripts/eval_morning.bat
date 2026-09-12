@echo off
REM Morning eval: QA held-out + CoT held-out. Mid-run: use step_N\final.dsb
setlocal
cd /d D:\TaoVm
set TAO_TOKENIZER=D:\TaoVm\build\tok_qa.bbp
set TAO_CPU_THREADS=8
set DSB=%1
if "%DSB%"=="" (
  if exist build\L1_qa_cot\final.dsb (set DSB=build\L1_qa_cot\final.dsb) else (
    for /f "delims=" %%i in ('dir /b /ad /o-n build\L1_qa_cot\step_* 2^>nul') do (
      if exist build\L1_qa_cot\%%i\final.dsb set DSB=build\L1_qa_cot\%%i\final.dsb & goto :got
    )
  )
)
:got
if "%DSB%"=="" (
  echo FAIL no checkpoint under build\L1_qa_cot
  exit /b 1
)
echo EVAL DSB=%DSB%
call scripts\eval_qa.bat %DSB%
node scripts\eval_cot.mjs %DSB% data\qa_cot\heldout_cot.jsonl
echo EXITCODE_EVAL_MORNING=%errorlevel%
