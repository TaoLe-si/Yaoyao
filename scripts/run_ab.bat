@echo off
cd /d D:\TaoVm
echo CTRL_START %TIME% > build\ab_status.txt
build\train_ctrl.exe build\noffn_l6\train.bin build\formal_tokenizer.bbp build\ab_ctrl 600 32 32
echo CTRL_DONE %TIME% >> build\ab_status.txt
build\train_l0.exe build\noffn_l6\train.bin build\formal_tokenizer.bbp build\ab_l0 600 32 32
echo L0_DONE %TIME% >> build\ab_status.txt
