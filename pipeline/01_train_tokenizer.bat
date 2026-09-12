@echo off
cd /d D:\TaoVm
set TAO_ALLOW_TOKENIZER=1
echo TOK2_START %TIME% > build\p1_tok_status.txt
build\bpe_train.exe D:\TaoVm\build\tok_v2.bbp 16121 90 D:\TaoVm\data\p1_reason.txt D:\TaoVm\data\alpaca_zh\conversations.txt E:\taovm-data\wiki_docs.txt E:\taovm-data\code_docs.txt >> build\p1_tok_status.txt 2>&1
echo EXIT=%errorlevel% >> build\p1_tok_status.txt
echo TOK2_DONE %TIME% >> build\p1_tok_status.txt
