@echo off
cd /d D:\TaoVm
del build\tile.log 1>nul 2>nul
build\profile_tiled.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\tile.log 2>&1 & build\profile_t16.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\tile.log 2>&1 & build\profile_t32.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\tile.log 2>&1 & build\profile_tiled.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\tile.log 2>&1 & build\profile_t16.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\tile.log 2>&1 & build\profile_t32.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\tile.log 2>&1
