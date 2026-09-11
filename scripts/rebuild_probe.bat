@echo off
cd /d D:\TaoVm
del build\PROBE_REBUILD_DONE.txt 2>nul
del build\probe_rebuild.log 2>nul
echo PROBE_REBUILD_START %TIME% > build\probe_rebuild.log
call scripts\build_noffn_probe2.bat >> build\probe_rebuild.log 2>&1
echo probe2_exit=%errorlevel% >> build\probe_rebuild.log
call scripts\build_noffn_probe_fast.bat >> build\probe_rebuild.log 2>&1
echo probe_fast_exit=%errorlevel% >> build\probe_rebuild.log
rmdir /s /q build\verify_def 2>nul
del build\verify_def.log 2>nul
build\train_noffn_probe2.exe build\noffn_probe3\train.bin build\formal_tokenizer.bbp build\verify_def 3 > nul 2>&1
echo verify_exit=%errorlevel% >> build\probe_rebuild.log
echo PROBE_REBUILD_DONE %TIME% >> build\probe_rebuild.log
echo done > build\PROBE_REBUILD_DONE.txt
