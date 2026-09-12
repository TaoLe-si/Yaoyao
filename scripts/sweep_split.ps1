# 决定性扫描：固定位置总数，比较 slots/width 的不同切分（GEMM batch = slots）。
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"
$OUT="D:\TaoVm\build\sweep_split.log"; Set-Content $OUT "fixed positions, vary slots/width split" -Encoding utf8
Get-Process train_shards* -EA SilentlyContinue | Stop-Process -Force; Start-Sleep -Seconds 3
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
foreach($cfg in @(@(3200,8,8),@(3200,16,4),@(3200,32,2),@(3200,64,1),@(2048,16,8),@(2048,32,4),@(2048,64,2),@(2048,64,1),@(2048,32,8),@(2048,128,1))){
  $d=$cfg[0];$sl=$cfg[1];$wd=$cfg[2]
  Set-Item Env:\TAO_CFG_D $d; Set-Item Env:\TAO_CFG_S ([string]([int]($d/2)))
  Set-Item Env:\TAO_CFG_M $d; Set-Item Env:\TAO_CFG_DK ([string]([int]($d/8)))
  $dir="D:\TaoVm\build\sp_tmp"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log","$dir.err","$dir.out" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"8","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0;$dl=(Get-Date).AddSeconds(360)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){$m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits);if($m -gt $peak){$peak=$m};Start-Sleep -Milliseconds 200}
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds;$tot=0;$n=0;$nll=@()
  if(Test-Path "$dir.log"){foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){
    if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){$tot+=[long]$matches[1];$n++;if($ln -match "NLL=([0-9.]+)"){$nll+=[double]$matches[1]}} }}
  if($n -ge 2){$l=("d={0,5} sl={1,3} wd={2,2} pos={3,4} t/step={4,6:N2}s tok/s={5,6:N0} vram={6,5}MiB NLL={7:N2}->{8:N2}" -f $d,$sl,$wd,($sl*$wd),($dt/$n),($tot/$dt),($peak-$base),$nll[0],$nll[-1])}
  else{$e="";if(Test-Path "$dir.log"){$e=(Get-Content "$dir.log"|Select-String "FAIL|not ready|error"|Select-Object -First 1).Line};$l=("d={0,5} sl={1,3} wd={2,2} pos={3,4} FAIL ({4:N1}s) {5}" -f $d,$sl,$wd,($sl*$wd),$dt,$e)}
  Write-Output $l; Add-Content $OUT $l -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue }
Write-Output "SWEEP_SPLIT_DONE"; Add-Content $OUT "SWEEP_SPLIT_DONE" -Encoding utf8
