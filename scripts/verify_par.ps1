# 验证：delta内核并行化 + 优化器状态页锁定，d=3200 与 d=1024
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"; $env:TAO_STAGE_TIMING="1"
$OUT="D:\TaoVm\build\vpar.log"; Set-Content $OUT "parallel kernels + pinned opt state" -Encoding utf8
Get-Process train_shards* -EA SilentlyContinue | Stop-Process -Force
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
foreach($cfg in @(@(3200,400,8,8),@(1024,128,16,16))){
  $d=$cfg[0];$dk=$cfg[1];$sl=$cfg[2];$wd=$cfg[3]
  Set-Item Env:\TAO_CFG_D $d; Set-Item Env:\TAO_CFG_S ([string]([int]$d/2))
  Set-Item Env:\TAO_CFG_M $d; Set-Item Env:\TAO_CFG_DK $dk
  $dir="D:\TaoVm\build\vp_$($d)_$($sl)"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"10","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards_blas.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0;$dl=(Get-Date).AddSeconds(900)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){$m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits);if($m -gt $peak){$peak=$m};Start-Sleep -Milliseconds 250}
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds;$tot=0;$n=0;$g=0.0;$u=0.0;$pj=0.0;$nll=@()
  if(Test-Path "$dir.log"){foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){
    if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){$tot+=[long]$matches[1];$n++;if($ln -match "NLL=([0-9.]+)"){$nll+=[double]$matches[1]}}
    if($ln -match "ms_graph=([0-9.]+) ms_upd=([0-9.]+) ms_project=([0-9.]+)"){$g+=[double]$matches[1];$u+=[double]$matches[2];$pj+=[double]$matches[3]} }}
  if($n -ge 2){
    $l=("OK   d={0,5} dk={1,4} sl={2,2} wd={3,2} t/步={4,5:N2}s tok/s={5,7:N0} ms_graph={6,6:N0} ms_upd={7,6:N0} ms_project={8,5:N0} vram={9,5}MiB NLL={10:N2}->{11:N2}" -f $d,$dk,$sl,$wd,($dt/$n),($tot/$dt),($g/$n),($u/$n),($pj/$n),($peak-$base),$nll[0],$nll[-1])
  }else{
    $msg="";foreach($f in @("$dir.log","$dir.err")){if(-not $msg -and (Test-Path $f)){$msg=(Get-Content $f -EA SilentlyContinue|Select-String "FAIL|error"|Select-Object -First 1).Line}}
    $l=("FAIL d={0} sl={1} {2} {3:N1}s" -f $d,$sl,$msg,$dt)}
  Write-Output $l; Add-Content $OUT $l -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue }
Write-Output "VPAR_DONE"; Add-Content $OUT "VPAR_DONE" -Encoding utf8
