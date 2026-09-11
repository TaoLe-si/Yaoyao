@echo off
cd /d D:\TaoVm
del build\REBUILD_DONE.txt 2>nul
del build\rebuild.log 2>nul
echo REBUILD_START %TIME% > build\rebuild.log
echo --- v01 (production) --- >> build\rebuild.log
call scripts\build_v01.bat >> build\rebuild.log 2>&1
echo v01_exit=%errorlevel% >> build\rebuild.log
echo --- delta_mem --- >> build\rebuild.log
call scripts\build_delta_fast.bat >> build\rebuild.log 2>&1
echo delta_exit=%errorlevel% >> build\rebuild.log
echo REBUILD_DONE %TIME% >> build\rebuild.log
echo done > build\REBUILD_DONE.txt
