@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul
if errorlevel 1 (echo VCVARS_FAIL & exit /b 1)
echo ---- graph variant ---- > build\graph_build.log
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_CPU_AVX2 -DTAO_ASYNC_ALLOC -DTAO_ASYNC_D2D -DTAO_DEFER_BACKWARD_SYNC -Xcompiler "/arch=AVX2 /O2" -I src src\train_shards.cu -o build\train_shards_graph.exe >> build\graph_build.log 2>&1
echo EXITCODE_GRAPH=%errorlevel% >> build\graph_build.log
echo GRAPH_BUILD_DONE >> build\graph_build.log
