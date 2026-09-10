@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
if not exist build mkdir build
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /I src src\benchmark_noffn_decode.cpp /Febuild\benchmark_noffn_decode.exe /Fobuild\benchmark_noffn_decode.obj bcrypt.lib
if errorlevel 1 exit /b 1
exit /b %errorlevel%
