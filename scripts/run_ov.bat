@echo off
cd /d D:\TaoVm
del build\OV_DONE.txt build\ov.log 2>nul
echo START > build\ov.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
set TAO_POOL_SPIN=1
set TAO_LAYER_LIMIT=2
echo CFG t1L2 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t1L4 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t1L6 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t1L8 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t8L2 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t8L4 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t8L6 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t8L8 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t1L2 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t1L4 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t1L6 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t1L8 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t8L2 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t8L4 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t8L6 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t8L8 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t1L2 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t1L4 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t1L6 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t1L8 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 1 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t8L2 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t8L4 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t8L6 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t8L8 >> build\ov.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\ov.log 2>&1
echo DONE >> build\ov.log
echo done > build\OV_DONE.txt
