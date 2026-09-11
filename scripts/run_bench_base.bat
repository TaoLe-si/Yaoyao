@echo off
cd /d D:\TaoVm
rmdir /s /q build\bench_base 2>nul
del build\bench_base.log 2>nul
del build\STOP_PROBE 2>nul
echo BENCH_BASE_START %TIME% >> build\arch_launch.txt
build\train_noffn_probe2.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\bench_base 300 > nul 2>&1
echo BENCH_BASE_DONE %TIME% >> build\arch_launch.txt
