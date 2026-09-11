@echo off
cd /d D:\TaoVm
del build\rep.log 1>nul 2>nul
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\rep.log 2>&1 & build\profile_tiled.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\rep.log 2>&1 & build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\rep.log 2>&1 & build\profile_tiled.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\rep.log 2>&1 & build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\rep.log 2>&1 & build\profile_tiled.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 24 1 1>>build\rep.log 2>&1
