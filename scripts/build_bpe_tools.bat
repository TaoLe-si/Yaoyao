@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /std:c++17 /EHsc /DNOMINMAX /I src src\bpe_train.cpp /Fe:build\bpe_train.exe /Fo:build\ 2>&1
echo A=%errorlevel%
cl /nologo /O2 /std:c++17 /EHsc /DNOMINMAX /I src src\tok_verify.cpp /Fe:build\tok_verify.exe /Fo:build\ 2>&1
echo B=%errorlevel%
