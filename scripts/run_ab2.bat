@echo off
cd /d D:\TaoVm
set TAO_ALLOW_TOKENIZER=1
echo AB_START %TIME% > build\ab2_status.txt
echo --- A: train_delta_long 600 steps --- >> build\ab2_status.txt
build\train_delta_long.exe D:\TaoVm\data\stage_core\train.bin D:\TaoVm\build\tok_digit_v1.bbp D:\TaoVm\build\ab2_old 600 32 32 >> build\ab2_status.txt 2>&1
echo A_EXIT=%errorlevel% >> build\ab2_status.txt
echo --- B: train_shards 603 steps (9 shards x 67) --- >> build\ab2_status.txt
build\train_shards.exe D:\TaoVm\data\cur_core D:\TaoVm\build\tok_digit_v1.bbp D:\TaoVm\build\ab2_new 67 32 32 >> build\ab2_status.txt 2>&1
echo B_EXIT=%errorlevel% >> build\ab2_status.txt
echo AB_DONE %TIME% >> build\ab2_status.txt
