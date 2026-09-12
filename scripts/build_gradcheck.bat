@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul
if errorlevel 1 (echo VCVARS_FAIL & exit /b 1)
if not exist build mkdir build
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -I src -DTAO_DELTA_MEM src\test_delta_grad.cu -o build\test_delta_grad.exe
echo EXITCODE_GRADCHECK=%errorlevel%
