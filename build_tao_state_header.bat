@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
cd /d D:/TaoVm
cl /nologo /EHsc /std:c++17 /O2 test_tao_state_header.cpp /Fe:test_tao_state_header.exe /link bcrypt.lib
if errorlevel 1 exit /b 1
test_tao_state_header.exe
