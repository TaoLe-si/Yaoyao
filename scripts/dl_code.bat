@echo off
cd /d D:\TaoVm
set HTTPS_PROXY=http://127.0.0.1:7890
set HTTP_PROXY=http://127.0.0.1:7890
set HF_ENDPOINT=https://hf-mirror.com
echo DL_START %TIME% > build\dl_status.txt
hf download codeparrot/codeparrot-clean-valid --repo-type dataset --local-dir E:\taovm-data\codeparrot-valid > build\dl_code.log 2>&1
echo EXIT=%errorlevel% >> build\dl_status.txt
echo DL_DONE %TIME% >> build\dl_status.txt
