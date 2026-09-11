@echo off
cd /d D:\TaoVm
set TAO_ALLOW_TOKENIZER=1
set TAO_LR=0.0001
set TAO_SHUFFLE_SEED=20260914
echo S2_START %TIME% > build\s2_status.txt
build\train_shards.exe D:\TaoVm\data\stage2_shards D:\TaoVm\build\tok_real_v1.bbp D:\TaoVm\build\train_s2 675 32 32 D:\TaoVm\build\s1_real_p2 0 >> build\s2_status.txt 2>&1
echo EXIT=%errorlevel% >> build\s2_status.txt
