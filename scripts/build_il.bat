@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_HAS_AVX512 /DTAO_NO_FFN /I src src\bench_interleaved.cpp /Febuild\bench_interleaved.exe /Fobuild\bench_interleaved.obj bcrypt.lib
