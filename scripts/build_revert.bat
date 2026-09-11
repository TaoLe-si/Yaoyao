@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
echo === TRAINER ===
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -I src src\train_noffn_probe.cu -o build\train_delta_long.exe
echo T=%errorlevel%
echo === CPU DECODER (reference) ===
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\cpu_resident_greedy_pipeline.cpp /Febuild\h2r_cpu_rev.exe /Fobuild\h2r_cpu_rev.obj bcrypt.lib
echo D=%errorlevel%
echo === DIAG ===
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\decode_diag.cpp /Febuild\decode_diag.exe /Fobuild\decode_diag.obj bcrypt.lib
echo G=%errorlevel%
