@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul
cl /nologo /O2 /EHsc /Fe"D:\TaoVm\.dsh-analysis\cpuid_probe.exe" /Fo"D:\TaoVm\.dsh-analysis\cpuid_probe.obj" "D:\TaoVm\.dsh-analysis\cpuid_probe.cpp"
if errorlevel 1 exit /b 1
"D:\TaoVm\.dsh-analysis\cpuid_probe.exe"
