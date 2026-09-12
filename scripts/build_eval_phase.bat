@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
cl /nologo /O2 /EHsc /std:c++17 /DNOMINMAX /arch:AVX512 /DTAO_CPU_AVX2 /DTAO_HAS_AVX512 /DTAO_NO_FFN /DTAO_DELTA_MEM /DTAO_PHASE_TIMING /I src src\eval_nll.cpp /Febuild\eval_phase.exe /Fobuild\eval_phase.obj bcrypt.lib 2>&1
echo EXITCODE_PHASE=%errorlevel%
