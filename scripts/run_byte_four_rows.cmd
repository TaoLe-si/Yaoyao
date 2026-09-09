@echo off
setlocal
cd /d "%~dp0.."
rem CPU-only, bounded benchmark. Run from an x64 Visual Studio developer prompt.
where cl >nul 2>nul
if errorlevel 1 (echo ERROR: use an x64 Visual Studio developer command prompt & exit /b 1)
rem Preserve all preexisting executables and logs, including previous candidates.
if exist build\benchmark_byte_four_rows.exe (echo ERROR: candidate executable already exists & exit /b 1)
if exist build\benchmark_byte_four_rows.obj (echo ERROR: candidate object already exists & exit /b 1)
if exist build\benchmark_byte_four_rows.log (echo ERROR: candidate log already exists & exit /b 1)
cl /nologo /I src /O2 /std:c++17 /arch:AVX2 /fp:strict /EHsc src\benchmark_byte_four_rows.cpp /Febuild\benchmark_byte_four_rows.exe /Fobuild\benchmark_byte_four_rows.obj
if errorlevel 1 exit /b 1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
build\benchmark_byte_four_rows.exe > build\benchmark_byte_four_rows.log 2>&1
exit /b %errorlevel%
