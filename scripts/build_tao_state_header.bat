@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
cd /d "%~dp0.."
if not exist build mkdir build
cl /nologo /I src /EHsc /std:c++17 /O2 src\test_tao_state_header.cpp /Fo:build\test_tao_state_header.obj /Fe:build\test_tao_state_header.exe /link bcrypt.lib
if errorlevel 1 exit /b 1
build\test_tao_state_header.exe
