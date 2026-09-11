@echo off
cd /d D:\TaoVm
set TAO_ALLOW_TOKENIZER=1
set TAO_LR=0.0003
echo S1R2_START %TIME% > build\s1r2_status.txt
build\train_shards.exe D:\TaoVm\data\cur_alpaca D:\TaoVm\build\tok_real_v1.bbp D:\TaoVm\build\s1_real_r1 400 32 32 D:\TaoVm\build\train_s1_real 4 >> build\s1r2_status.txt 2>&1
echo EXIT=%errorlevel% >> build\s1r2_status.txt
echo S1R2_DONE %TIME% >> build\s1r2_status.txt
