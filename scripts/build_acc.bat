@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 src\microbench_accumulators.cpp /Febuild\microbench_accumulators.exe /Fobuild\microbench_accumulators.obj
