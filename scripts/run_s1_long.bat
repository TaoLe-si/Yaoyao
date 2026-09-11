@echo off
cd /d D:\TaoVm
set TAO_ALLOW_TOKENIZER=1
echo S1L_START %TIME% > build\s1l_status.txt
build\train_shards.exe D:\TaoVm\data\cur_core D:\TaoVm\build\tok_digit_v1.bbp D:\TaoVm\build\train_s1_long 200 32 32 >> build\s1l_status.txt 2>&1
echo EXIT=%errorlevel% >> build\s1l_status.txt
echo S1L_DONE %TIME% >> build\s1l_status.txt
