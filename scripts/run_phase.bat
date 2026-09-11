@echo off
cd /d D:\TaoVm
del build\PHASE_DONE.txt build\phase.log 2>nul
echo START %TIME% > build\phase.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
echo === R3+R4 threads 1,2,4,8,16 === >> build\phase.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1,2,4,8,16 1 >> build\phase.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
echo === R0 threads 1,2,4,8,16 (mincost=1) === >> build\phase.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1,2,4,8,16 1 >> build\phase.log 2>&1
echo DONE %TIME% >> build\phase.log
echo done > build\PHASE_DONE.txt
