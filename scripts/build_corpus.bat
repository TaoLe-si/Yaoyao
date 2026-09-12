@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /I src src\build_corpus.cpp /Febuild\build_corpus.exe /Fobuild\build_corpus.obj bcrypt.lib
echo EXITCODE_BUILD_CORPUS=%errorlevel%
