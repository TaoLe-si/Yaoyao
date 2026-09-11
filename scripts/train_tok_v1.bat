@echo off
cd /d D:\TaoVm
echo TOK_START %TIME% > build\tok_status.txt
& "D:\TaoVm\build\bpe_train.exe" "D:\TaoVm\build\tok_digit_v1.bbp" "16123" "90" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "D:\TaoVm\data\rebalanced_v2\conversations.txt" "E:\taovm-data\code_docs.txt" > build\tok_train.log 2>&1
echo EXIT=%errorlevel% >> build\tok_status.txt
echo TOK_DONE %TIME% >> build\tok_status.txt
