@echo off
cd /d D:\TaoVm
echo === FRESH ENV ===
echo CUDA_PATH=%CUDA_PATH%
where nvcc
echo === VCVARSALL ===
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
where cl
echo === NVCC VERSION ===
nvcc --version
echo === COMPILE ===
nvcc -O2 -std=c++17 -arch=sm_89 -DNOMINMAX -DTAO_DELTA_MEM --default-stream per-thread -I src src\train_noffn_probe.cu -o build\probe_offload_test.exe
echo EXITCODE=%errorlevel%
