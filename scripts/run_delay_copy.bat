@echo off
cd /d D:\TaoVm
echo DELAY_START %TIME% > build\delay_status.txt
build\train_delay.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\delay_copy 400 32 32
echo DELAY_DONE %TIME% >> build\delay_status.txt
