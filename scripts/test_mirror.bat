@echo off
set HF_ENDPOINT=https://hf-mirror.com
cd /d D:\TaoVm
hf download codeparrot/codeparrot-clean-valid --repo-type dataset --local-dir E:\taovm-data\_conn_test > E:\taovm-data\mirror_test.txt 2>&1
echo EXITCODE=%errorlevel% >> E:\taovm-data\mirror_test.txt
