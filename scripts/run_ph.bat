@echo off
cd /d D:\TaoVm
del build\PH_DONE.txt build\ph.log 2>nul
echo START %TIME% > build\ph.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
set TAO_POOL_SPIN=1
echo CFG best >> build\ph.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ph.log 2>&1
echo CFG r0 >> build\ph.log
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ph.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
echo CFG best >> build\ph.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ph.log 2>&1
echo CFG r0 >> build\ph.log
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ph.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
echo CFG best >> build\ph.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ph.log 2>&1
echo CFG r0 >> build\ph.log
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ph.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
echo DONE >> build\ph.log
echo done > build\PH_DONE.txt
