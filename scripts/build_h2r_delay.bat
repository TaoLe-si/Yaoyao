@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\cpu_resident_greedy_pipeline.cpp /Febuild\h2r_delay.exe /Fobuild\h2r_delay.obj bcrypt.lib
echo EXITCODE=%errorlevel%
