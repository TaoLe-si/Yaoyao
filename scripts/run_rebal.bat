@echo off
cd /d D:\TaoVm
echo REBAL_START %TIME% > build\rebal_status.txt
build\train_delta_long.exe build\rebalanced\train.bin build\formal_tokenizer.bbp build\rebal 900 32 32
echo REBAL_DONE %TIME% >> build\rebal_status.txt
