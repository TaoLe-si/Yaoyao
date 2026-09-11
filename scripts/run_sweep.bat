@echo off
cd /d D:\TaoVm
echo SWEEP_LAUNCH %DATE% %TIME% > build\sweep_launch.txt
for %%D in (sweep_base sweep_s512 sweep_s1024 sweep_L4 sweep_dk128) do if exist build\%%D rmdir /s /q build\%%D
if exist build\sweep_base.log del build\sweep_base.log
if exist build\sweep_s512.log del build\sweep_s512.log
if exist build\sweep_s1024.log del build\sweep_s1024.log
if exist build\sweep_L4.log del build\sweep_L4.log
if exist build\sweep_dk128.log del build\sweep_dk128.log

set TAO_CFG_S=128
set TAO_CFG_LAYERS=2
set TAO_CFG_DK=64
start "sweep_base" /B cmd /c "build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sweep_base 800 32 32 > build\sweep_base.stdout.txt 2>&1"

set TAO_CFG_S=512
start "sweep_s512" /B cmd /c "build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sweep_s512 800 32 32 > build\sweep_s512.stdout.txt 2>&1"

set TAO_CFG_S=1024
start "sweep_s1024" /B cmd /c "build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sweep_s1024 800 32 32 > build\sweep_s1024.stdout.txt 2>&1"

set TAO_CFG_S=128
set TAO_CFG_LAYERS=4
start "sweep_L4" /B cmd /c "build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sweep_L4 800 32 32 > build\sweep_L4.stdout.txt 2>&1"

set TAO_CFG_LAYERS=2
set TAO_CFG_DK=128
start "sweep_dk128" /B cmd /c "build\train_delta_long.exe build\copy_order\train.bin build\formal_tokenizer.bbp build\sweep_dk128 800 32 32 > build\sweep_dk128.stdout.txt 2>&1"

echo LAUNCHED_ALL %TIME% >> build\sweep_launch.txt
