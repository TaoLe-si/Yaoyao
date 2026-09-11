@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX512 src\microbench_avx512.cpp /Febuild\microbench_avx512.exe /Fobuild\microbench_avx512.obj
