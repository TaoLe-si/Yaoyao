@echo off
cd /d D:\TaoVm
set TAO_ALLOW_TOKENIZER=1
echo S1R_START %TIME% > build\s1r_status.txt
build\train_shards.exe D:\TaoVm\data\cur_alpaca D:\TaoVm\build\tok_real_v1.bbp D:\TaoVm\build\train_s1_real 400 32 32 >> build\s1r_status.txt 2>&1
echo EXIT=%errorlevel% >> build\s1r_status.txt
echo S1R_DONE %TIME% >> build\s1r_status.txt
