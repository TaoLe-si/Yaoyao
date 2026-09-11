@echo off
cd /d D:\TaoVm
del build\BOTTLENECK_DONE.txt build\bottleneck.log 2>nul
echo BOTTLENECK_START %TIME% > build\bottleneck.log
echo === R0 identity, threads 2,4,8 === >> build\bottleneck.log
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2,4,8 >> build\bottleneck.log 2>&1
echo === R3 layershare, threads 2,4,8 === >> build\bottleneck.log
set TAO_LAYER_SHARE=0,0,0,0,4,4,4,4
build\benchmark_noffn_s3.exe build\arch_A_ds3\step_150\final.dsb 2,4,8 >> build\bottleneck.log 2>&1
echo BOTTLENECK_DONE %TIME% >> build\bottleneck.log
echo done > build\BOTTLENECK_DONE.txt
