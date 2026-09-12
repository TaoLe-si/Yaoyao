@echo off
cd /d D:\TaoVm\build
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 bench_vocab_cache.cpp /Febench_vocab_cache.exe >nul 2>&1
bench_vocab_cache.exe
