@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /std:c++17 /EHsc /DNOMINMAX /I src src\taovm_pipe.cpp /Fe:build\taovm_pipe.exe /Fo:build\ 2>&1
echo EXITCODE=%errorlevel%
