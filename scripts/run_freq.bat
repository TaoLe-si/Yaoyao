@echo off
cd /d D:\TaoVm
del build\FREQ_DONE.txt build\freq.log 2>nul
echo START > build\freq.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
set TAO_POOL_SPIN=1
echo --- SOLO n=1 --- >> build\freq.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\freq.log 2>&1
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\freq.log 2>&1
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\freq.log 2>&1
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
start /min "" cmd /c "scripts\burn.bat"
ping -n 4 127.0.0.1 >nul
echo --- LOADED n=1 --- >> build\freq.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\freq.log 2>&1
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\freq.log 2>&1
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\freq.log 2>&1
echo --- LOADED n=8 --- >> build\freq.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\freq.log 2>&1
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\freq.log 2>&1
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\freq.log 2>&1
taskkill /IM cmd.exe /F /FI "PID ne %PM%" 1>nul 2>nul
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq *" 1>nul 2>nul
echo DONE >> build\freq.log
echo done > build\FREQ_DONE.txt
