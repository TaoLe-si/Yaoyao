@echo off
cd /d D:\TaoVm\build
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 probe_sample_equiv.cpp /Feprobe_sample_equiv.exe
if errorlevel 1 echo COMPILE_FAILED
if exist probe_sample_equiv.exe probe_sample_equiv.exe
