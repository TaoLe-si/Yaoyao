@echo off
cd /d D:\TaoVm
set TAO_CFG_DK=64
set TAO_CFG_LAYERS=2
set TAO_CFG_S=128
echo BASE_START %TIME% > build\sweep_status.txt
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sw2_base 300 32 32
echo BASE_DONE %TIME% >> build\sweep_status.txt
set TAO_CFG_S=512
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sw2_s512 300 32 32
echo S512_DONE %TIME% >> build\sweep_status.txt
set TAO_CFG_S=128
set TAO_CFG_LAYERS=4
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sw2_L4 300 32 32
echo L4_DONE %TIME% >> build\sweep_status.txt
echo ALL_DONE %TIME% >> build\sweep_status.txt
