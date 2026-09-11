@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
echo === FFN TRAINER ===
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_WITH_FFN -I src src\train_noffn_probe.cu -o build\train_h2r_ffn.exe
echo EXITCODE_TRAIN=%errorlevel%
echo === FFN DECODER ===
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_DELTA_MEM /I src src\decode_diag.cpp /Febuild\decode_diag_ffn.exe /Fobuild\decode_diag_ffn.obj bcrypt.lib
echo EXITCODE_DEC=%errorlevel%
