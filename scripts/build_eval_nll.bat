@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DNOMINMAX /DTAO_CPU_AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\eval_nll.cpp /Febuild\eval_nll.exe /Fobuild\eval_nll.obj bcrypt.lib
echo EXITCODE_EVAL_NLL=%errorlevel%
