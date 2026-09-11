@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\cpu_resident_greedy_pipeline.cpp /Febuild\h2r_cpu.exe /Fobuild\h2r_cpu.obj bcrypt.lib
