@echo off
cd /d D:\TaoVm
del build\SPIN_DONE.txt build\spin.log 2>nul
echo START %TIME% > build\spin.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
set TAO_POOL_SPIN=
rem
echo CFG sleep >> build\spin.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,16 >> build\spin.log 2>&1
set TAO_POOL_SPIN=
set TAO_POOL_SPIN=1
echo CFG spin >> build\spin.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,16 >> build\spin.log 2>&1
set TAO_POOL_SPIN=
rem
echo CFG sleep >> build\spin.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,16 >> build\spin.log 2>&1
set TAO_POOL_SPIN=
set TAO_POOL_SPIN=1
echo CFG spin >> build\spin.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,16 >> build\spin.log 2>&1
set TAO_POOL_SPIN=
rem
echo CFG sleep >> build\spin.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,16 >> build\spin.log 2>&1
set TAO_POOL_SPIN=
set TAO_POOL_SPIN=1
echo CFG spin >> build\spin.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4,8,16 >> build\spin.log 2>&1
echo DONE %TIME% >> build\spin.log
echo done > build\SPIN_DONE.txt
