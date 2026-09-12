@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /DNOMINMAX /arch:AVX512 /DTAO_CPU_AVX2 /DTAO_HAS_AVX512 /I src src\ternary_pack_test.cpp /Febuild\ternary_pack_test.exe /Fobuild\ternary_pack_test.obj
echo EXITCODE_TPT=%errorlevel%
