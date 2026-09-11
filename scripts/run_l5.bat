@echo off
rem L5 泛化修正：导出 -> 训练 H2R 与 ds3 -> 验收
rem 用法: scripts\run_l5.bat [UPDATES]   默认 400
setlocal
cd /d "%~dp0.."
set UPDATES=%1
if "%UPDATES%"=="" set UPDATES=400

if not exist build\noffn_l5 mkdir build\noffn_l5

echo [1/4] 生成 L5 语料
node scripts\gen_corpus_l5.mjs D:/TaoVm/data/noffn_l5
if errorlevel 1 exit /b 1

echo [2/4] 导出 train.bin
if exist build\noffn_l5\train.bin (
  echo     already exported, skip
) else (
  build\export_noffn_probe.exe data\noffn_l5\conversations.txt build\noffn_l5\train.bin build\formal_tokenizer.bbp
  if errorlevel 1 exit /b 1
)

echo [3/4] 训练（ds3 与 H2R 各 %UPDATES% 步）
if exist build\ds3_l5_run (
  echo     ds3_l5_run exists, skip
) else (
  build\train_noffn_probe.exe build\noffn_l5\train.bin build\formal_tokenizer.bbp build\ds3_l5_run %UPDATES%
  if errorlevel 1 exit /b 1
)
if exist build\h2r_l5_run (
  echo     h2r_l5_run exists, skip
) else (
  build\train_noffn_probe_delta.exe build\noffn_l5\train.bin build\formal_tokenizer.bbp build\h2r_l5_run %UPDATES%
  if errorlevel 1 exit /b 1
)

echo [4/4] 验收（判据见 doc 17 1.4）
set TAO_CPU_EXE=D:\TaoVm\build\h2r_cpu.exe
node scripts\eval_binding_indist.mjs H2R_L5=build\h2r_l5_run\step_%UPDATES%\final.dsb
node scripts\eval_copy_fidelity.mjs H2R_L5=build\h2r_l5_run\step_%UPDATES%\final.dsb
set TAO_CPU_EXE=D:\TaoVm\build\yaoyao_cpu_v01.exe
node scripts\eval_binding_indist.mjs ds3_L5=build\ds3_l5_run\step_%UPDATES%\final.dsb
node scripts\eval_copy_fidelity.mjs ds3_L5=build\ds3_l5_run\step_%UPDATES%\final.dsb
exit /b 0
