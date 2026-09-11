@echo off
cd /d D:\TaoVm
echo TERM_START %TIME% > build\term_status.txt
build\train_delta_long.exe build\rebalanced_term\train.bin build\formal_tokenizer.bbp build\rebal_term 900 32 32
echo TERM_DONE %TIME% >> build\term_status.txt
