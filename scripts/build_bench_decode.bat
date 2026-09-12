@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX512 /DTAO_HAS_AVX512 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\bench_decode_tps.cpp /Febuild\bench_decode_tps.exe /Fobuild\bench_decode_tps.obj bcrypt.lib
echo EXITCODE=%errorlevel%
