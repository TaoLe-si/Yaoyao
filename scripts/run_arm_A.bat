@echo off
cd /d D:\TaoVm
rmdir /s /q build\arch_A_ds3 1>nul 2>nul
del build\arch_A_ds3.log 1>nul 2>nul
build\train_noffn_probe_fast.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\arch_A_ds3 700 32 32 > build\A.out 2>&1
echo DONE_A %DATE% %TIME% >> build\arch_launch.txt
