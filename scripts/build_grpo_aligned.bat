@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
rem Single build path: every trainer links cuBLAS. Measured bit-identical to the
rem hand-written per-slot GEMM (NLL delta 0.000E+000) and 19 percent faster.
rem The old non-BLAS branch has been deleted.
echo ---- grpo_rollout ----
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /I src src\grpo_rollout.cpp /Febuild\grpo_rollout.exe /Fobuild\grpo_rollout.obj bcrypt.lib
echo EXITCODE_ROLLOUT=%errorlevel%
echo ---- train_grpo ----
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_GPU_HEALTH -DTAO_BATCH_CUBLAS -DTAO_CPU_AVX2 -Xcompiler "/arch=AVX2 /O2" -I src src\train_grpo.cu -o build\train_grpo.exe -lcublas
echo EXITCODE_TRAIN_GRPO=%errorlevel%
echo ---- train_sft ----
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_GPU_HEALTH -DTAO_BATCH_CUBLAS -DTAO_CPU_AVX2 -Xcompiler "/arch=AVX2 /O2" -I src src\train_sft.cu -o build\train_sft.exe -lcublas
echo EXITCODE_TRAIN_SFT=%errorlevel%
echo ---- train_shards ----
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_GPU_HEALTH -DTAO_BATCH_CUBLAS -DTAO_CPU_AVX2 -Xcompiler "/arch=AVX2 /O2" -I src src\train_shards.cu -o build\train_shards.exe -lcublas
echo EXITCODE_TRAIN_SHARDS=%errorlevel%
echo ---- taovm_pipe ----
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_GPU_HEALTH -DTAO_CPU_AVX2 -Xcompiler "/arch=AVX2 /O2" -I src src\taovm_pipe.cpp -o build\taovm_pipe.exe
echo EXITCODE_PIPE=%errorlevel%
echo ALL_BUILDS_DONE
