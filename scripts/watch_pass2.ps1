$ErrorActionPreference="Continue"
cd D:\TaoVm
'WATCH2_START ' + (Get-Date -Format HH:mm:ss) | Add-Content D:\TaoVm\build\p2_status.txt
while (Get-Process -Name train_shards -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
if (-not (Test-Path 'D:\TaoVm\build\s1_real_r2\opt_state.bin')) { 'P2_ABORT no opt_state' | Add-Content D:\TaoVm\build\p2_status.txt; exit 1 }
// 第一轮已退出，exe 解锁，先重编译（带种子参数），再启动第二轮
& D:\TaoVm\scripts\build_train_shards.bat *>> D:\TaoVm\build\p2_status.txt
if ($LASTEXITCODE -ne 0) { 'P2_ABORT build fail' | Add-Content D:\TaoVm\build\p2_status.txt; exit 1 }
$env:TAO_ALLOW_TOKENIZER='1'
$env:TAO_LR='0.0001'
$env:TAO_SHUFFLE_SEED='20260913'
'P2_START ' + (Get-Date -Format HH:mm:ss) + ' seed=20260913' | Add-Content D:\TaoVm\build\p2_status.txt
& D:\TaoVm\build\train_shards.exe D:\TaoVm\data\cur_alpaca D:\TaoVm\build\tok_real_v1.bbp D:\TaoVm\build\s1_real_p2 400 32 32 D:\TaoVm\build\s1_real_r2 0 *>> D:\TaoVm\build\p2_status.txt
'P2_EXIT=' + $LASTEXITCODE | Add-Content D:\TaoVm\build\p2_status.txt
