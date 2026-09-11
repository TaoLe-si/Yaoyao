@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_HAS_AVX512 /DTAO_NO_FFN /I src src\benchmark_noffn_s3.cpp /Febuild\benchmark_noffn_s3.exe /Fobuild\benchmark_noffn_s3.obj bcrypt.lib
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /DTAO_NO_PHASE_TIMING /I src src\benchmark_noffn_s3.cpp /Febuild\benchmark_noffn_s3_plain.exe /Fobuild\benchmark_noffn_s3_plain.obj bcrypt.lib
