@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX512 /DNOMINMAX /DTAO_CPU_AVX2 /DTAO_HAS_AVX512 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\grpo_rollout.cpp /Febuild\grpo_rollout.exe /Fobuild\grpo_rollout.obj bcrypt.lib
echo EXITCODE_ROLLOUT=%errorlevel%
