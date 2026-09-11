@echo off
cd /d D:\TaoVm
echo L6_START %TIME% > build\l6_status.txt
build\train_delay.exe build\noffn_l6\train.bin build\formal_tokenizer.bbp build\delay_l6 600 32 32
echo L6_DONE %TIME% >> build\l6_status.txt
