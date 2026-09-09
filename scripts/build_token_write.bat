@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul
if errorlevel 1 exit /b 1
if not exist build mkdir build
for %%S in (optimize_token_write test_token_write_hard inspect_token_gate benchmark_write_gate_scale) do (
 cl /nologo /I src /EHsc /O2 /std:c++17 /arch:AVX2 %%S.cpp /Fo:build\%%S.obj /Fe:build\%%S.exe
 if errorlevel 1 exit /b 1
)
rem Run optimize_token_write.exe SEED adam before test_token_write_hard.exe SEED.
