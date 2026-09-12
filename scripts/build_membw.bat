@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 src\membw_probe.cpp /Febuild\membw_probe.exe /Fobuild\membw_probe.obj
echo EXITCODE_MEMBW=%errorlevel%
