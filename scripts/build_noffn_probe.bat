@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
if not exist build mkdir build
cl /nologo /O2 /EHsc /std:c++17 /I src src\export_noffn_probe.cpp /Febuild\export_noffn_probe.exe /Fobuild\export_noffn_probe.obj bcrypt.lib
if errorlevel 1 exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -I src src\train_noffn_probe.cu -o build\train_noffn_probe.exe
exit /b %errorlevel%
