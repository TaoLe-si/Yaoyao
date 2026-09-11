@echo off
cd /d D:\TaoVm
del build\sweep.log 1>nul 2>nul
echo === slots=4 width=128 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 4 128 2 1>>build\sweep.log 2>&1
echo === slots=4 width=64 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 4 64 2 1>>build\sweep.log 2>&1
echo === slots=4 width=48 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 4 48 2 1>>build\sweep.log 2>&1
echo === slots=8 width=48 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 8 48 2 1>>build\sweep.log 2>&1
echo === slots=16 width=48 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 16 48 2 1>>build\sweep.log 2>&1
echo === slots=24 width=48 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 24 48 2 1>>build\sweep.log 2>&1
echo === slots=32 width=48 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 32 48 2 1>>build\sweep.log 2>&1
echo === slots=16 width=32 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 16 32 2 1>>build\sweep.log 2>&1
echo === slots=32 width=32 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 32 32 2 1>>build\sweep.log 2>&1
echo === slots=48 width=32 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 48 32 2 1>>build\sweep.log 2>&1
echo === slots=64 width=32 === 1>>build\sweep.log
build\profile_probe.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp 64 32 2 1>>build\sweep.log 2>&1
echo SWEEP_DONE 1>>build\sweep.log
