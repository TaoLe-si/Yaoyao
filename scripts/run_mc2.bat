@echo off
cd /d D:\TaoVm
del build\MC2_DONE.txt build\mc2.log 2>nul
echo START %TIME% > build\mc2.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
set TAO_POOL_SPIN=1
echo CFG spin1 >> build\mc2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,12,16 1 >> build\mc2.log 2>&1
echo CFG spin1 >> build\mc2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,12,16 1 >> build\mc2.log 2>&1
echo CFG spin1 >> build\mc2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,12,16 1 >> build\mc2.log 2>&1
set TAO_POOL_SPIN=
echo CFG sleep1 >> build\mc2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,12,16 1 >> build\mc2.log 2>&1
echo DONE >> build\mc2.log
echo done > build\MC2_DONE.txt
