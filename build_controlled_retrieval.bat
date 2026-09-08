@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
if not exist build mkdir build
cl /nologo /EHsc /O2 /std:c++17 make_controlled_retrieval.cpp /Fo:build/make_controlled_retrieval.obj /Fe:build/make_controlled_retrieval.exe
if errorlevel 1 exit /b 1
for %%S in (train_controlled_readout search_controlled_gates controlled_retrieval_controls verify_controlled_readout) do (
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" %%S.cu -o build/%%S.exe -allow-unsupported-compiler --fmad=false -O2 -std=c++17 -arch=sm_75 -lbcrypt
    if errorlevel 1 exit /b 1
)
