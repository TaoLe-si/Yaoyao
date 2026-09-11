@echo off
cd /d D:\TaoVm
set TAO_ALLOW_TOKENIZER=1
set TAO_LR=0.0001
echo S1R3_START %TIME% > build\s1r3_status.txt
build\train_shards.exe D:\TaoVm\data\cur_alpaca D:\TaoVm\build\tok_real_v1.bbp D:\TaoVm\build\s1_real_r2 400 32 32 D:\TaoVm\build\s1_real_r1 6 >> build\s1r3_status.txt 2>&1
echo EXIT=%errorlevel% >> build\s1r3_status.txt
echo S1R3_DONE %TIME% >> build\s1r3_status.txt
