@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
nvcc -O2 -std=c++17 -arch=sm_89 -DNOMINMAX -DTAO_DELTA_MEM -I src src\act_parity_test.cu -o build\act_parity_test.exe
echo BUILD_EXITCODE=%errorlevel%
