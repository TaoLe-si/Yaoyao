# 优化器状态驻内存（主机映射）、前向权重驻显存：测吞吐与显存
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="2"
$OUT="D:\TaoVm\build\hwopt.log"; Set-Content $OUT ("opt-mapped " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($d,$sl,$wd,$dk){
  $script:i++
  Set-Item Env:\TAO_CFG_D $d; Set-Item Env:\TAO_CFG_S ([string]([int]$d/2))
  Set-Item Env:\TAO_CFG_M $d; Set-Item Env:\TAO_CFG_DK $dk
  $dir="D:\TaoVm\build\o_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"5","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $sat=@(); $dl=(Get-Date).AddSeconds(900)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}
    $sat+=[int](nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    Start-Sleep -Milliseconds 200 }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds; $tot=0;$n=0
  if(Test-Path "$dir.log"){foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){$tot+=[long]$matches[1];$n++}}}
  $avg=if($sat.Count){[math]::Round(($sat|Measure-Object -Average).Average,1)}else{0}
  if($n -ge 1){
    $nll="-"; if((Get-Content "$dir.log" -EA SilentlyContinue|Select-String "train_preupdate_NLL=([0-9.]+)"|Select-Object -Last 1)){$nll=$matches[1]}
    $line=("OK   d={0,5} dk={1,3} sl={2,2} wd={3,2} t/步={4,5:N2}s tok/s={5,7:N0} GPU={6,5}% vram={7,5}MiB NLL={8}" -f $d,$dk,$sl,$wd,($dt/$n),($tot/$dt),$avg,($peak-$base),$nll)
  }else{
    $msg=""; foreach($f in @("$dir.log","$dir.err")){if(-not $msg -and (Test-Path $f)){$msg=(Get-Content $f -EA SilentlyContinue|Select-String "FAIL"|Select-Object -First 1).Line}}
    $short=if($msg -match "out of memory"){"OOM"}elseif($msg -match "not ready"){"NOTREADY"}else{"OTHER"}
    $line=("FAIL d={0,5} dk={1,3} sl={2,2} wd={3,2} {4} {5:N1}s vram峰值={6,5}MiB" -f $d,$dk,$sl,$wd,$short,$dt,($peak-$base)) }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue }
B 1024 16 16 128
B 3200  8  8 400
B 3200 16  8 400
B 3200  8 16 400
B 3200 16 16 400
Write-Output "OPT_DONE"; Add-Content $OUT "OPT_DONE" -Encoding utf8
