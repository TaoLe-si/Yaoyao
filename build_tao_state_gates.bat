@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
cd /d "%~dp0"
if not exist build mkdir build
cl /nologo /EHsc /std:c++17 /O2 /openmp eval_tao_state_gates.cpp /Fo:build/eval_tao_state_gates.obj /Fe:build/eval_tao_state_gates.exe /link bcrypt.lib
if errorlevel 1 exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" train_tao_state_gates.cu -o build/train_tao_state_gates.exe -allow-unsupported-compiler --fmad=false -O2 -std=c++17 -arch=sm_75 -lcublas -lbcrypt
