@echo off
chcp 65001 >nul
echo ==========================================
echo   夭夭 v21 编译 + 运行
echo ==========================================

cd /d D:\TaoVm

echo.
echo [1/3] 检查编译器...
where clang++ >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 clang++
    echo 请安装 LLVM: https://llvm.org/
    pause
    exit /b 1
)

echo [2/3] 编译 yaoyao_v21_full.cpp ...
"C:\Program Files\LLVM\bin\clang++.exe" -O2 -mavx2 -mfma -fopenmp -o yaoyao_v21_full.exe yaoyao_v21_full.cpp
if errorlevel 1 (
    echo 编译失败!
    pause
    exit /b 1
)
echo   ✓ 编译成功

echo.
echo [3/3] 运行训练...
echo.
yaoyao_v21_full.exe D:\TaoVm\tinystories_train.txt 1024 500 2 0.005

echo.
echo ==========================================
echo   训练完成
echo ==========================================
pause
