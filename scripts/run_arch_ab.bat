@echo off
cd /d D:\TaoVm
rmdir /s /q build\arch_A_ds3 1>nul 2>nul
del build\arch_A_ds3.log 1>nul 2>nul
rmdir /s /q build\arch_B_h2r 1>nul 2>nul
del build\arch_B_h2r.log 1>nul 2>nul
echo START_A %TIME% >> build\arch_launch.txt
build\train_noffn_probe_fast.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\arch_A_ds3 250 32 32 > build\A.out 2>&1
echo DONE_A %TIME% >> build\arch_launch.txt
echo START_B %TIME% >> build\arch_launch.txt
build\train_delta_fast.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\arch_B_h2r 1000 32 20 > build\B.out 2>&1
echo DONE_B %TIME% >> build\arch_launch.txt
