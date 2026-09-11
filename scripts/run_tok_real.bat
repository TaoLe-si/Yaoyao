@echo off
cd /d D:\TaoVm
set TAO_ALLOW_TOKENIZER=1
echo TOK_START %TIME% > build\tok2_status.txt
build\bpe_train.exe D:\TaoVm\build\tok_real_v1.bbp 16123 90 D:\TaoVm\data\alpaca_zh\conversations.txt E:\taovm-data\code_docs.txt >> build\tok2_status.txt 2>&1
echo EXIT=%errorlevel% >> build\tok2_status.txt
echo TOK_DONE %TIME% >> build\tok2_status.txt
