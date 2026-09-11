@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /EHsc /std:c++17 src\microbench_string_map.cpp /Febuild\microbench_string_map.exe /Fobuild\microbench_string_map.obj
