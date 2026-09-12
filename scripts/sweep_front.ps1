# 找 512 位置（64 单位）是否落在 WDDM 悬崖之下，以及 60 单位的表现。
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_STAGE_TIMING="1"; $env:TAO_CFG_LAYERS="2"
$env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"; $env:TAO_CFG_DK="400"
$OUT="D:\TaoVm\build\sweep_front.log"; Set-Content $OUT "frontier: 64 and 60 units at d=3200" -Encoding utf8
Get-Process train_shards* -EA SilentlyContinue | Stop-Process -Force; Start-Sleep -Seconds 3
foreach($cfg in @(@(16,4),@(20,3),@(8,8),@(10,6))){
  $sl=$cfg[0];$wd=$cfg[1]
  Remove-Item -Recurse -Force build\fr_tmp -EA SilentlyContinue; Remove-Item -Force build\fr_tmp.log -EA SilentlyContinue
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList @("build\probe_one","build\tok_v2.bbp","build\fr_tmp","10","$sl","$wd") -PassThru -NoNewWindow
  $peak=0;$dl=(Get-Date).AddSeconds(200)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){$m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}; Start-Sleep -Milliseconds 120}
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $pos=0;$n=0;$g=@();$u=@()
  if(Test-Path build\fr_tmp.log){foreach($ln in (Get-Content build\fr_tmp.log)){ if($ln -match "^UPDATE"){$n++; if($ln -match "positions=(\d+)"){$pos+=[long]$matches[1]}; if($ln -match "ms_graph=([0-9.]+)"){$g+=[double]$matches[1]}; if($ln -match "ms_upd=([0-9.]+)"){$u+=[double]$matches[1]} } }}
  if($n -ge 5){ $g2=($g|Select-Object -Skip 2|Measure-Object -Average).Average; $u2=($u|Select-Object -Skip 2|Measure-Object -Average).Average; $l=("sl={0,3} wd={1,2} units={2,3} peak={3,5}MiB ms_graph={4,6:N0} ms_upd={5,6:N0} ss_tok/s={6,6:N0}" -f $sl,$wd,($sl*$wd),$peak,$g2,$u2,(($sl*$wd*8)/(($g2+$u2)/1000))) }
  else { $l=("sl={0,3} wd={1,2} units={2,3} FAIL ({3:N1}s) peak={4}MiB" -f $sl,$wd,($sl*$wd),$dt,$peak) }
  Write-Output $l; Add-Content $OUT $l -Encoding utf8
  Remove-Item -Recurse -Force build\fr_tmp -EA SilentlyContinue }
Write-Output "SWEEP_FRONT_DONE"; Add-Content $OUT "SWEEP_FRONT_DONE" -Encoding utf8
