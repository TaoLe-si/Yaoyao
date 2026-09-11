@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 src\microbench_engram_lookup.cpp /Febuild\microbench_engram_lookup.exe /Fobuild\microbench_engram_lookup.obj
