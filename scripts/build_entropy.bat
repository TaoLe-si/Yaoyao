@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /DNOMINMAX /arch:AVX2 /I src src\entropy_ref.cpp /Febuild\entropy_ref.exe /Fobuild\entropy_ref.obj 2>&1
echo EXITCODE_ENT=%errorlevel%
