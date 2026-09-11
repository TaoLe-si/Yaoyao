@echo off
cd /d D:\TaoVm
del build\OV2_DONE.txt build\ov2.log 2>nul
echo START > build\ov2.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
set TAO_POOL_SPIN=1
set TAO_LAYER_LIMIT=2
echo CFG t2L2 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t2L4 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t2L6 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t2L8 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t4L2 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t4L4 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t4L6 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t4L8 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t2L2 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t2L4 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t2L6 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t2L8 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t4L2 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t4L4 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t4L6 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t4L8 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t2L2 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t2L4 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t2L6 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t2L8 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG t4L2 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG t4L4 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG t4L6 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG t4L8 >> build\ov2.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 1 >> build\ov2.log 2>&1
echo DONE >> build\ov2.log
echo done > build\OV2_DONE.txt
