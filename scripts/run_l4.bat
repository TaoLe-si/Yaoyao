@echo off
rem L4 语料（18038 篇）2 层泛化：导出 -> 重编 probe -> 训练 -> 评测
rem 用法: scripts\run_l4.bat [UPDATES]   默认 400
setlocal
cd /d "%~dp0.."
set UPDATES=%1
if "%UPDATES%"=="" set UPDATES=400

if not exist build mkdir build
if not exist build\noffn_l4 mkdir build\noffn_l4

echo [1/4] export train.bin
if exist build\noffn_l4\train.bin (
  echo     already exported, skip
) else (
  build\export_noffn_probe.exe data\noffn_l4\conversations.txt build\noffn_l4\train.bin build\formal_tokenizer.bbp
  if errorlevel 1 exit /b 1
)

echo [2/4] rebuild 2-layer probe trainer
call scripts\build_noffn_probe.bat
if errorlevel 1 exit /b 1

echo [3/4] train %UPDATES% updates
if exist build\noffn_l4_run (
  echo     output exists; refusing to overwrite
  exit /b 1
)
build\train_noffn_probe.exe build\noffn_l4\train.bin build\formal_tokenizer.bbp build\noffn_l4_run %UPDATES%
if errorlevel 1 exit /b 1

echo [4/4] evaluate
for %%S in (50 100 200 300 400) do (
  if exist build\noffn_l4_run\step_%%S\final.dsb (
    echo --- binding step %%S ---
    build\diag_binding_rate.exe build\noffn_l4_run\step_%%S\final.dsb
  )
)
node scripts\eval_probe3.mjs L4=build\noffn_l4_run\final.dsb
exit /b 0
