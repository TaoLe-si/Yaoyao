@echo off
chcp 65001 >nul
cd /d D:\TaoVm
echo 编译 yaoyao_v21_full.cpp ...
"C:\Program Files\LLVM\bin\clang++.exe" -O2 -mavx2 -mfma -fopenmp -o yaoyao_v21_full.exe yaoyao_v21_full.cpp
if errorlevel 1 (
    echo 编译失败
) else (
    echo 编译成功: yaoyao_v21_full.exe
)
pause
