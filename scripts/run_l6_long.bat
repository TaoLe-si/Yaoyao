@echo off
cd /d D:\TaoVm
echo LAUNCH %DATE% %TIME% > build\long_launch.txt
build\train_delta_long.exe build\noffn_l6\train.bin build\formal_tokenizer.bbp build\h2r_l6_long 3000 32 32
echo EXITCODE=%errorlevel% >> build\long_launch.txt
