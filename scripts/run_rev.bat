@echo off
cd /d D:\TaoVm
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\rev_new 2 32 32
echo DONE %TIME% > build\rev_status.txt
