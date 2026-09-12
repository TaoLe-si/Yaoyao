# ncu：跳过初始化，只统计训练区内核的累计时长
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"
$env:TAO_CFG_D="1024"; $env:TAO_CFG_S="512"; $env:TAO_CFG_M="1024"; $env:TAO_CFG_DK="128"
$dir="D:\TaoVm\build\ncu2"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
$ncu="C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\ncu.bat"
$ar=@("--metrics","gpu__time_duration.sum","--launch-skip","900","--launch-count","600","--target-processes","all","--csv","build\train_shards.exe","build\probe_one","build\tok_v2.bbp",$dir,"3","8","8")
& $ncu @ar 2>&1 | Out-File -Encoding utf8 build\ncu_dur.log
Write-Output "NCUDUR_DONE"
