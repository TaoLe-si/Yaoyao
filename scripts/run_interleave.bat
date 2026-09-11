@echo off
cd /d D:\TaoVm
del build\INTERLEAVE_DONE.txt build\interleave.log 2>nul
echo START %TIME% > build\interleave.log
set TAO_LAYER_SHARE=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,0,0,0,0
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,0,0,0,0
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,0,0,0,0
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,0,0,0,0
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,0,0,0,0
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
set TAO_LAYER_SHARE=0,0,0,0,0,0,0,0
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\interleave.log 2>&1
echo DONE %TIME% >> build\interleave.log
echo done > build\INTERLEAVE_DONE.txt
