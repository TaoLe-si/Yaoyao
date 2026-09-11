@echo off
cd /d D:\TaoVm
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
echo === CTRL (no delay) ===
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_NO_DELAY -I src src\train_noffn_probe.cu -o build\train_ctrl.exe
echo T_CTRL=%errorlevel%
cl /nologo /O2 /EHsc /std:c++17 /arch=AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /DTAO_NO_DELAY /I src src\cpu_resident_greedy_pipeline.cpp /Febuild\h2r_ctrl.exe /Fobuild\h2r_ctrl.obj bcrypt.lib
echo D_CTRL=%errorlevel%
echo === L0ONLY ===
nvcc -O2 -std=c++17 -arch=sm_89 --default-stream per-thread -DNOMINMAX -DTAO_DELTA_MEM -DTAO_DELAY_L0 -I src src\train_noffn_probe.cu -o build\train_l0.exe
echo T_L0=%errorlevel%
cl /nologo /O2 /EHsc /std:c++17 /arch=AVX2 /DTAO_NO_FFN /DTAO_DELTA_MEM /DTAO_DELAY_L0 /I src src\cpu_resident_greedy_pipeline.cpp /Febuild\h2r_l0.exe /Fobuild\h2r_l0.obj bcrypt.lib
echo D_L0=%errorlevel%
