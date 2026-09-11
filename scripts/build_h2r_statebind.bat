@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\diag_state_binding.cpp /Febuild\h2r_statebind.exe /Fobuild\h2r_statebind.obj bcrypt.lib
exit /b %errorlevel%
