@echo off
setlocal
cd /d "%~dp0.."
if exist build\noffn_fresh (echo Existing training directory retained. & exit /b 1)
if exist build\noffn_fresh.log (echo Existing log retained. & exit /b 1)
if not exist build\STOP_TRAINING (echo Missing required STOP_TRAINING invariant. & exit /b 1)
if exist build\STOP_NOFFN (echo Pause request exists. & exit /b 1)
build\yaoyao_train_v01.exe build unused build\repair_export_20260909_v2\train.bin build\repair_export_20260909_v2\train.bin.accepted 7ce86dd8cce85ab1b3c2c8ea63b170a4b121c489ee0a5f3e1cac2f5fb1a13f8c build\noffn_fresh 2000
