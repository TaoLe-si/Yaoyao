@echo off
cd /d D:\TaoVm
echo V2_START %TIME% > build\v2_status.txt
build\train_delta_long.exe build\rebalanced_v2\train.bin build\formal_tokenizer.bbp build\rebal_v2 900 32 32
echo EXIT=%errorlevel% >> build\v2_status.txt
echo V2_DONE %TIME% >> build\v2_status.txt
