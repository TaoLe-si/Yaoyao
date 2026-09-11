@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\test_h2r_consistency.cpp /Febuild\test_h2r_consistency.exe /Fobuild\test_h2r_consistency.obj bcrypt.lib
