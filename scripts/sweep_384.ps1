# 同为 384 位置（slots*width=48）下的切分对比：GEMM batch = slots。
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_STAGE_TIMING="1"; $env:TAO_CFG_LAYERS="2"
$env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"; $env:TAO_CFG_DK="400"
$OUT="D:\TaoVm\build\sweep_384.log"; Set-Content $OUT "384 positions: vary slots/width split" -Encoding utf8
Get-Process train_shards* -EA SilentlyContinue | Stop-Process -Force; Start-Sleep -Seconds 3
foreach($cfg in @(@(8,6),@(12,4),@(16,3),@(24,2),@(48,1))){
  $sl=$cfg[0];$wd=$cfg[1]
  Remove-Item -Recurse -Force build\s384_tmp -EA SilentlyContinue; Remove-Item -Force build\s384_tmp.log -EA SilentlyContinue
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList @("build\probe_one","build\tok_v2.bbp","build\s384_tmp","8","$sl","$wd") -PassThru -NoNewWindow
  $peak=0;$dl=(Get-Date).AddSeconds(180)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){$m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}; Start-Sleep -Milliseconds 120}
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $pos=0;$n=0;$g=@()
  if(Test-Path build\s384_tmp.log){foreach($ln in (Get-Content build\s384_tmp.log)){ if($ln -match "^UPDATE"){$n++; if($ln -match "positions=(\d+)"){$pos+=[long]$matches[1]}; if($ln -match "ms_graph=([0-9.]+)"){$g+=[double]$matches[1]} } }}
  if($n -ge 3){ $tps=$pos/$dt; $g2=($g|Select-Object -Skip 1|Measure-Object -Average).Average; $l=("sl={0,3} wd={1,2} steps={2} peak={3,5}MiB t/step={4,5:N2}s tok/s={5,6:N1} ms_graph={6,6:N0}" -f $sl,$wd,$n,$peak,($dt/$n),$tps,$g2) }
  else { $l=("sl={0,3} wd={1,2} FAIL ({2:N1}s) peak={3}MiB" -f $sl,$wd,$dt,$peak) }
  Write-Output $l; Add-Content $OUT $l -Encoding utf8
  Remove-Item -Recurse -Force build\s384_tmp -EA SilentlyContinue }
Write-Output "SWEEP_384_DONE"; Add-Content $OUT "SWEEP_384_DONE" -Encoding utf8
