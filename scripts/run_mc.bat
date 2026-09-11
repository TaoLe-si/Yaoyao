@echo off
cd /d D:\TaoVm
del build\MC_DONE.txt build\mc.log 2>nul
echo START %TIME% > build\mc.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
set TAO_POOL_SPIN=1
echo CFG mc262144 >> build\mc.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 262144 >> build\mc.log 2>&1
echo CFG mc16384 >> build\mc.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 16384 >> build\mc.log 2>&1
echo CFG mc1 >> build\mc.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\mc.log 2>&1
echo CFG mc262144 >> build\mc.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 262144 >> build\mc.log 2>&1
echo CFG mc16384 >> build\mc.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 16384 >> build\mc.log 2>&1
echo CFG mc1 >> build\mc.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\mc.log 2>&1
echo CFG mc262144 >> build\mc.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 262144 >> build\mc.log 2>&1
echo CFG mc16384 >> build\mc.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 16384 >> build\mc.log 2>&1
echo CFG mc1 >> build\mc.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\mc.log 2>&1
echo CFG R3only_mc262144 >> build\mc.log
set TAO_LAYER_REUSE_MV=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 262144 >> build\mc.log 2>&1
set TAO_LAYER_REUSE_MV=1
echo CFG R3only_mc16384 >> build\mc.log
set TAO_LAYER_REUSE_MV=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 16384 >> build\mc.log 2>&1
set TAO_LAYER_REUSE_MV=1
echo DONE >> build\mc.log
echo done > build\MC_DONE.txt
