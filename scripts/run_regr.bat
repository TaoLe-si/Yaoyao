@echo off
cd /d D:\TaoVm
echo OLD_START %TIME% > build\regr_status.txt
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\regr_old 2 32 32
echo OLD_DONE %TIME% >> build\regr_status.txt
build\train_delay.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\regr_new 2 32 32
echo NEW_DONE %TIME% >> build\regr_status.txt
