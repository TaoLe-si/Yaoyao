@echo off
cd /d D:\TaoVm
del build\bench_new.txt build\bench_old.txt 2>nul
:wait
if exist build\bench_base\final.dsb goto run
timeout /t 20 /nobreak >nul
goto wait
:run
build\diag_bench.exe build\bench_base\final.dsb data\bench_v1\spec.tsv > build\bench_new.txt 2>&1
echo NEW_DONE %TIME% >> build\arch_launch.txt
build\diag_bench.exe build\arch_A_ds3\step_150\final.dsb data\bench_v1\spec.tsv > build\bench_old.txt 2>&1
echo BENCH_EVAL_DONE %TIME% >> build\arch_launch.txt
