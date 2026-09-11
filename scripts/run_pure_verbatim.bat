@echo off
cd /d D:\TaoVm
echo PV_START %TIME% > build\pv_status.txt
build\train_delta_long.exe build\pv_data\train.bin build\formal_tokenizer.bbp build\pure_verbatim 600 32 32
echo EXIT=%errorlevel% >> build\pv_status.txt
echo PV_DONE %TIME% >> build\pv_status.txt
