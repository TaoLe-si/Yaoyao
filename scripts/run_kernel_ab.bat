@echo off
cd /d D:\TaoVm
rmdir /s /q build\kern_base build\kern_fast 2>nul
del build\kern_base.log build\kern_fast.log 2>nul
del build\STOP_PROBE 2>nul
echo KERN_AB_START %TIME% >> build\arch_launch.txt
build\train_noffn_probe2.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\kern_base 6 4 128 > nul 2>&1
echo BASE_DONE %TIME% >> build\arch_launch.txt
build\train_noffn_probe_fast.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\kern_fast 6 4 128 > nul 2>&1
echo FAST_DONE %TIME% >> build\arch_launch.txt
