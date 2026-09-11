@echo off
cd /d D:\TaoVm
rmdir /s /q build\arch_B_h2r 1>nul 2>nul
del build\arch_B_h2r.log 1>nul 2>nul
build\train_delta_fast.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\arch_B_h2r 700 32 16 > build\B.out 2>&1
echo DONE_B %DATE% %TIME% >> build\arch_launch.txt
