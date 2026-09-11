@echo off
cd /d D:\TaoVm
echo SEQ_LAUNCH %DATE% %TIME% > build\sweep_launch.txt
for %%D in (sweep_base sweep_s512 sweep_L4) do if exist build\%%D rmdir /s /q build\%%D
for %%F in (sweep_base sweep_s512 sweep_L4) do if exist build\%%F.log del build\%%F.log

set TAO_CFG_LAYERS=2
set TAO_CFG_DK=64
set TAO_CFG_S=128
echo === base s=128 === >> build\sweep_launch.txt
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sweep_base 400 32 32

set TAO_CFG_S=512
echo === s512 === >> build\sweep_launch.txt
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sweep_s512 400 32 32

set TAO_CFG_S=128
set TAO_CFG_LAYERS=4
echo === L4 === >> build\sweep_launch.txt
build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sweep_L4 400 32 32

echo SEQ_DONE %TIME% >> build\sweep_launch.txt
