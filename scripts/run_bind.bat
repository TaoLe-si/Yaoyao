@echo off
cd /d D:\TaoVm
set TAO_ALLOW_TOKENIZER=1
echo BIND_START %TIME% > build\bind_status.txt
build\train_shards.exe D:\TaoVm\data\cur_bind D:\TaoVm\build\tok_digit_v1.bbp D:\TaoVm\build\train_bind 150 32 32 >> build\bind_status.txt 2>&1
echo EXIT=%errorlevel% >> build\bind_status.txt
echo BIND_DONE %TIME% >> build\bind_status.txt
