@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_CPU_AVX2 -Xcompiler "/arch:AVX2 /O2" -I src src\train_shards.cu -o build\train_shards_w.exe 2>&1
echo EXITCODE=%errorlevel%
