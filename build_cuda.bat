@echo off
set CUDAHOSTCXX=C:\Program Files\LLVM\bin\clang++.exe
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" -O2 -arch=sm_89 --allow-unsupported-compiler --no-cuda-host-compiler -o "D:\TaoVm\cuda_test.exe" "D:\TaoVm\cuda_test.cu"
