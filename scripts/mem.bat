@echo off
cd /d D:\TaoVm
del build\mem.log 1>nul 2>nul
echo === 4/128 === 1>>build\mem.log & build\profile_tiled.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 4 128 1 1>>build\mem.log 2>&1 & echo === 32/32 === 1>>build\mem.log & build\profile_tiled.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 32 32 1 1>>build\mem.log 2>&1 & echo === 64/24 === 1>>build\mem.log & build\profile_tiled.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\mem.log 2>&1 & echo === 96/24 === 1>>build\mem.log & build\profile_tiled.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 96 24 1 1>>build\mem.log 2>&1
