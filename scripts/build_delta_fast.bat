@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_SLOT_TILED_GEMM -I src src\train_noffn_probe.cu -o build\train_delta_fast.exe
exit /b %errorlevel%
