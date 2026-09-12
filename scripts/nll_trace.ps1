# 完整 NLL 轨迹：d=3200 dk=400 sl=8 wd=8，40 步，保留原始日志
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="2"; $env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"; $env:TAO_CFG_DK="400"
$dir="D:\TaoVm\build\nt"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
$ar=@("build\probe_one","build\tok_v2.bbp",$dir,"40","8","8")
& .\build\train_shards.exe @ar 2>&1 | Out-Null
Write-Output "=== UPDATE 轨迹 ==="
if(Test-Path "$dir.log"){ Get-Content "$dir.log" | Where-Object {$_ -match "^(UPDATE|SHARD_)"} } else { Write-Output "无日志" }
Write-Output "NLLTRACE_DONE"
