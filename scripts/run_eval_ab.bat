@echo off
cd /d D:\TaoVm
set TAO_CPU_EXE=D:\TaoVm\build\yaoyao_cpu_v01.exe
node scripts\eval_probe3.mjs A_crit=build/arch_A_ds3/step_100/final.dsb > build\eval_A.txt 2>&1
set TAO_CPU_EXE=D:\TaoVm\build\h2r_cpu.exe
node scripts\eval_probe3.mjs B_crit=build/arch_B_h2r/step_173/final.dsb > build\eval_B.txt 2>&1
echo EVAL_AB_DONE %TIME% >> build\arch_launch.txt
