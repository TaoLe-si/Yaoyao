@echo off
cd /d D:\TaoVm
del build\S2_DONE.txt build\spin2.log 2>nul
echo START %TIME% > build\spin2.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
set TAO_POOL_SPIN=
echo CFG sleep >> build\spin2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1,2,4,8,16 >> build\spin2.log 2>&1
set TAO_POOL_SPIN=1
echo CFG spin >> build\spin2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1,2,4,8,16 >> build\spin2.log 2>&1
set TAO_POOL_SPIN=
echo CFG sleep >> build\spin2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1,2,4,8,16 >> build\spin2.log 2>&1
set TAO_POOL_SPIN=1
echo CFG spin >> build\spin2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1,2,4,8,16 >> build\spin2.log 2>&1
set TAO_POOL_SPIN=
echo CFG sleep >> build\spin2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1,2,4,8,16 >> build\spin2.log 2>&1
set TAO_POOL_SPIN=1
echo CFG spin >> build\spin2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1,2,4,8,16 >> build\spin2.log 2>&1
echo DONE >> build\spin2.log
echo done > build\S2_DONE.txt
