@echo off
cd /d D:\TaoVm
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\copy_order_run 800 32 32
echo EXITCODE=%errorlevel% >> build\copy_launch.txt
