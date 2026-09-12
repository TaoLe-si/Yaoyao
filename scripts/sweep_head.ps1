# 找「刚好不触发 WDDM 分页」的批大小：峰值显存 < 7.4GB 时才有满速。
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_STAGE_TIMING="1"; $env:TAO_CFG_LAYERS="2"
$env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"; $env:TAO_CFG_DK="400"
$OUT="D:\TaoVm\build\sweep_head.log"; Set-Content $OUT "d=3200 batch size vs VRAM headroom" -Encoding utf8
Get-Process train_shards* -EA SilentlyContinue | Stop-Process -Force; Start-Sleep -Seconds 3
foreach($cfg in @(@(8,8),@(8,6),@(8,4),@(16,4),@(16,2),@(24,2),@(32,1),@(8,2))){
  $sl=$cfg[0];$wd=$cfg[1]
  Remove-Item -Recurse -Force build\hd_tmp -EA SilentlyContinue; Remove-Item -Force build\hd_tmp.log -EA SilentlyContinue
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList @("build\probe_one","build\tok_v2.bbp","build\hd_tmp","8","$sl","$wd") -PassThru -NoNewWindow
  $peak=0; while(-not $p.HasExited){$m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}; Start-Sleep -Milliseconds 120}
  $dt=((Get-Date)-$t0).TotalSeconds
  $pos=0;$n=0;$g=@()
  if(Test-Path build\hd_tmp.log){foreach($ln in (Get-Content build\hd_tmp.log)){ if($ln -match "^UPDATE"){$n++; if($ln -match "positions=(\d+)"){$pos+=[long]$matches[1]}; if($ln -match "ms_graph=([0-9.]+)"){$g+=[double]$matches[1]} } }}
  if($n -ge 3){ $tps=$pos/$dt; $g2=($g|Select-Object -Skip 1|Measure-Object -Average).Average; $l=("sl={0,3} wd={1,2} steps={2} peak={3,5}MiB t/step={4,5:N2}s tok/s={5,6:N1} ms_graph={6,6:N0}" -f $sl,$wd,$n,$peak,($dt/$n),$tps,$g2) }
  else { $l=("sl={0,3} wd={1,2} FAIL/OOM ({2:N1}s) peak={3}MiB" -f $sl,$wd,$dt,$peak) }
  Write-Output $l; Add-Content $OUT $l -Encoding utf8
  Remove-Item -Recurse -Force build\hd_tmp -EA SilentlyContinue }
Write-Output "SWEEP_HEAD_DONE"; Add-Content $OUT "SWEEP_HEAD_DONE" -Encoding utf8
