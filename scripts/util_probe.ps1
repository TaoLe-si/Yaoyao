$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
Remove-Item Env:\TAO_OPT_OFFLOAD -EA SilentlyContinue
$env:TAO_CFG_LAYERS="2"; $env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"; $env:TAO_CFG_DK="400"
Remove-Item -Recurse -Force build\util1 -EA SilentlyContinue
$p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList @("build\probe_one","build\tok_v2.bbp","build\util1","8","8","8") -PassThru -NoNewWindow -RedirectStandardOutput build\util1.o -RedirectStandardError build\util1.e
$u=@(); $dl=(Get-Date).AddSeconds(180)
while(-not $p.HasExited -and (Get-Date) -lt $dl){
  $s=(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits) -split ","
  $u += [PSCustomObject]@{G=[int]$s[0].Trim(); M=[int]$s[1].Trim(); V=[int]$s[2].Trim()}
  Start-Sleep -Milliseconds 250
}
if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
Write-Output ("样本数=" + $u.Count)
if($u.Count){
  Write-Output ("GPU 利用率: 均值=" + [math]::Round(($u|Measure-Object G -Average).Average,1) + "%  最大=" + ($u|Measure-Object G -Maximum).Maximum + "%")
  Write-Output ("显存带宽利用率: 均值=" + [math]::Round(($u|Measure-Object M -Average).Average,1) + "%  最大=" + ($u|Measure-Object M -Maximum).Maximum + "%")
  Write-Output ("显存占用: 最大=" + ($u|Measure-Object V -Maximum).Maximum + " MiB")
  $hi=($u|Where-Object {$_.G -ge 60}).Count
  Write-Output ("GPU>=60% 的样本占比 = " + [math]::Round(100.0*$hi/$u.Count,1) + "%")
}
Get-Content build\util1.log -EA SilentlyContinue | Select-String "^UPDATE" | Select-Object -First 4 | ForEach-Object { $_.Line }
