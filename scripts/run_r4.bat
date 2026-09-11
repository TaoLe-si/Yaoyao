@echo off
cd /d D:\TaoVm
del build\R4_DONE.txt build\r4.log 2>nul
echo START %TIME% > build\r4.log
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
echo CFG R0 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
echo CFG R3 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4& set TAO_LAYER_REUSE_MV=1
echo CFG R3+R4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=4
echo CFG L=4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=2
echo CFG L=2 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
echo CFG R0 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
echo CFG R3 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4& set TAO_LAYER_REUSE_MV=1
echo CFG R3+R4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=4
echo CFG L=4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=2
echo CFG L=2 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
echo CFG R0 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
echo CFG R3 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4& set TAO_LAYER_REUSE_MV=1
echo CFG R3+R4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=4
echo CFG L=4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=2
echo CFG L=2 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
echo CFG R0 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
echo CFG R3 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4& set TAO_LAYER_REUSE_MV=1
echo CFG R3+R4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=4
echo CFG L=4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=2
echo CFG L=2 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
echo CFG R0 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
echo CFG R3 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4& set TAO_LAYER_REUSE_MV=1
echo CFG R3+R4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=4
echo CFG L=4 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
set TAO_LAYER_SHARE=
set TAO_LAYER_REUSE_MV=
set TAO_LAYER_LIMIT=
set TAO_LAYER_LIMIT=2
echo CFG L=2 >> build\r4.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 4 >> build\r4.log 2>&1
echo DONE %TIME% >> build\r4.log
echo done > build\R4_DONE.txt
