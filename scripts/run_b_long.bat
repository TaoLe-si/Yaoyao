@echo off
cd /d D:\TaoVm
rmdir /s /q build\arch_B_long 1>nul 2>nul
del build\arch_B_long.log 1>nul 2>nul
del build\STOP_PROBE 1>nul 2>nul
echo START_BLONG %TIME% >> build\arch_launch.txt
build\train_delta_fast.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\arch_B_long 600 32 20 > build\Blong.out 2>&1
echo DONE_BLONG %TIME% >> build\arch_launch.txt
