@echo off
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "MSVC=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.44.35207"
set "PATH=%MSVC%\bin\Hostx64\x64;%CUDA%\bin;%PATH%"
set "INCLUDE="
set "LIB="
"%CUDA%\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_89 -ccbin "%MSVC%\bin\Hostx64\x64\cl.exe" build\nano.cu -o build\fix1.exe > build\fix1.exe.log 2>&1
echo EXITCODE=%errorlevel% >> build\fix1.exe.log
