@echo off
cd /d D:\TaoVm
echo CEIL_START %TIME% > build\ceil_status.txt
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\copy_ceil 1200 32 32
echo CEIL_DONE %TIME% >> build\ceil_status.txt
