@echo off
cd /d D:\TaoVm
del build\STOP_PROBE 1>nul 2>nul
rmdir /s /q build\arch_A_ds3 1>nul 2>nul
rmdir /s /q build\arch_B_h2r 1>nul 2>nul
del build\arch_A_ds3.log 1>nul 2>nul
del build\arch_B_h2r.log 1>nul 2>nul
del build\A.out 1>nul 2>nul
del build\B.out 1>nul 2>nul
echo LAUNCH %DATE% %TIME% > build\arch_launch.txt
start /b cmd /c "build\train_noffn_probe_fast.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\arch_A_ds3 700 32 32 > build\A.out 2>&1"
start /b cmd /c "build\train_delta_fast.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\arch_B_h2r 700 32 16 > build\B.out 2>&1"
exit /b 0
