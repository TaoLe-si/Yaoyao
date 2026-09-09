@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul
if errorlevel 1 exit /b 1
if not exist build mkdir build
for %%S in (train_raw_boundary optimize_raw_commit4 test_commit_gradient test_raw_stream_commit4 test_raw_stream_limits benchmark_staged_commit) do (
 cl /nologo /I src /EHsc /O2 /std:c++17 /arch:AVX2 %%S.cpp /Fo:build\%%S.obj /Fe:build\%%S.exe
 if errorlevel 1 exit /b 1
)
rem Train local diagnostic weights before running stream tests:
rem for %%S in (7 123 2026 42 999) do build\train_raw_boundary.exe %%S
rem for %%S in (7 123 2026 42 999) do build\optimize_raw_commit4.exe %%S adam 2
