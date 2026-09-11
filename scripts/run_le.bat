@echo off
cd /d D:\TaoVm
del build\LE_DONE.txt build\le.log 2>nul
echo START > build\le.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
set TAO_LAYER_REUSE_MV=1
set TAO_POOL_SPIN=1
set TAO_LAYER_LIMIT=
echo CFG L8 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG L2 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG L4 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG L6 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG L8 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG L2 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG L4 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG L6 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=
echo CFG L8 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=2
echo CFG L2 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=4
echo CFG L4 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
set TAO_LAYER_LIMIT=6
echo CFG L6 >> build\le.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 8 1 >> build\le.log 2>&1
echo DONE >> build\le.log
echo done > build\LE_DONE.txt
