@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_SLOT_TILED_GEMM -DTAO_SLOT_TILE=16 -I src src\profile_probe.cu -o build\profile_t16.exe
