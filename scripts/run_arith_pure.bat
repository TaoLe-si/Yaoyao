@echo off
cd /d D:\TaoVm
echo ARITH_START %TIME% > build\arith_status.txt
build\train_ctrl.exe build\arith_pure\train.bin build\formal_tokenizer.bbp build\arith_ctrl 600 32 32
echo ARITH_DONE %TIME% >> build\arith_status.txt
