@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
where cl
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -I src src\train_noffn_probe.cu -o build\train_delta_long.exe
echo EXITCODE=%errorlevel%
