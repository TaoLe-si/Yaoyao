@echo off
setlocal
call "D:\TaoVm\scripts\build_grpo_aligned.bat"
if errorlevel 1 (echo REBUILD_ALL_FAIL & exit /b 1)
echo REBUILD_ALL_DONE
