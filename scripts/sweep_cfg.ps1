# 配置扫描：在显存约束下找最大 tok/s。GEMM batch = slots（width 为串行）。
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"
$OUT="D:\TaoVm\build\sweep_cfg.log"; Set-Content $OUT "config sweep (BLAS, pinned opt state)" -Encoding utf8
Get-Process train_shards* -EA SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 800
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$combos=@()
foreach($d in @(2048,2560,3200)){ foreach($sl in @(8,16,24,32)){ foreach($wd in @(8,16,32)){ $combos+=[pscustomobject]@{d=$d;sl=$sl;wd=$wd} } } }
foreach($c in $combos){
  $d=$c.d;$sl=$c.sl;$wd=$c.wd
  Set-Item Env:\TAO_CFG_D $d; Set-Item Env:\TAO_CFG_S ([string]([math]::Round($d/2)))
  Set-Item Env:\TAO_CFG_M $d; Set-Item Env:\TAO_CFG_DK ([string]([math]::Round($d/8)))
  $dir="D:\TaoVm\build\sw_tmp"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"8","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards_blas.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0;$dl=(Get-Date).AddSeconds(240)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){$m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits);if($m -gt $peak){$peak=$m};Start-Sleep -Milliseconds 200}
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds;$tot=0;$n=0;$nll=@()
  if(Test-Path "$dir.log"){foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){
    if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){$tot+=[long]$matches[1];$n++;if($ln -match "NLL=([0-9.]+)"){$nll+=[double]$matches[1]}} }}
  if($n -ge 2){
    $l=("d={0,5} sl={1,3} wd={2,3} pos={3,5} t/步={4,5:N2}s tok/s={5,7:N0} vram={6,5}MiB NLL={7:N2}->{8:N2}" -f $d,$sl,$wd,($sl*$wd),($dt/$n),($tot/$dt),($peak-$base),$nll[0],$nll[-1])
  }else{ $l=("d={0,5} sl={1,3} wd={2,3} FAIL/OOM ({3:N1}s)" -f $d,$sl,$wd,$dt) }
  Write-Output $l; Add-Content $OUT $l -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue }
Write-Output "SWEEP_CFG_DONE"; Add-Content $OUT "SWEEP_CFG_DONE" -Encoding utf8
