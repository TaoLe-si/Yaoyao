# 干净的内核时长剖析：跳过初始化，按内核聚合累计耗时
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"
$env:TAO_CFG_D="1024"; $env:TAO_CFG_S="512"; $env:TAO_CFG_M="1024"; $env:TAO_CFG_DK="128"
Get-Process train_shards*,ncu -EA SilentlyContinue | Stop-Process -Force
$dir="D:\TaoVm\build\nc"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
$ncu="C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\ncu.bat"
$ar=@("--metrics","gpu__time_duration.sum","--launch-skip","1200","--launch-count","400","--target-processes","all","--csv","--log-file","D:\TaoVm\build\ncu_c.csv","build\train_shards.exe","build\probe_one","build\tok_v2.bbp",$dir,"3","8","8")
& $ncu @ar 2>&1 | Out-Null
Write-Output "NCUC_DONE"
