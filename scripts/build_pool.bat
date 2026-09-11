@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /I src src\microbench_pool.cpp /Febuild\microbench_pool.exe /Fobuild\microbench_pool.obj
