@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
if not exist build mkdir build
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -I src src\test_tao_ternary_cuda.cu -o build/test_tao_ternary_cuda.exe -allow-unsupported-compiler --fmad=false -O2 -std=c++17 -arch=sm_75
if errorlevel 1 exit /b 1
build\test_tao_ternary_cuda.exe
if errorlevel 1 exit /b 1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -I src src\test_tao_embedding_cuda.cu -o build/test_tao_embedding_cuda.exe -allow-unsupported-compiler --fmad=false -O2 -std=c++17 -arch=sm_75
if errorlevel 1 exit /b 1
build\test_tao_embedding_cuda.exe
