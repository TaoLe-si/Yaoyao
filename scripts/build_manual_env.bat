@echo off
setlocal
set "MSVC=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.51.36231"
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "SDKV=10.0.26100.0"
set "PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKV%\x64;%PATH%"
set "INCLUDE=%MSVC%\include;%SDK%\Include\%SDKV%\ucrt;%SDK%\Include\%SDKV%\um;%SDK%\Include\%SDKV%\shared;%SDK%\Include\%SDKV%\winrt"
set "LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKV%\ucrt\x64;%SDK%\Lib\%SDKV%\um\x64"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_89 -DNOMINMAX -DTAO_DELTA_MEM --default-stream per-thread -allow-unsupported-compiler -ccbin "%MSVC%\bin\Hostx64\x64\cl.exe" -I src src\train_noffn_probe.cu -o build\probe_offload_test.exe > build\offload_out.txt 2>&1
echo EXITCODE=%errorlevel% >> build\offload_out.txt
