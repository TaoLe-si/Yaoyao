# BLAS vs 非BLAS 对照，同一配置，含阶段计时
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"
$env:TAO_STAGE_TIMING="1"
$env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"; $env:TAO_CFG_DK="400"
$OUT="D:\TaoVm\build\blascmp.log"; Set-Content $OUT "blas compare" -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
foreach($exe in @("train_shards","train_shards_blas")){
  $dir="D:\TaoVm\build\bc_$exe"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"5","8","8")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\$exe.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0;$dl=(Get-Date).AddSeconds(600)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){$m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits);if($m -gt $peak){$peak=$m};Start-Sleep -Milliseconds 250}
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds;$tot=0;$n=0;$g=0.0;$u=0.0
  if(Test-Path "$dir.log"){foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){
    if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){$tot+=[long]$matches[1];$n++}
    if($ln -match "ms_graph=([0-9.]+) ms_upd=([0-9.]+)"){$g+=[double]$matches[1];$u+=[double]$matches[2]} }}
  if($n -ge 1){$l=("exe={0,-18} t/步={1,5:N2}s tok/s={2,7:N0} ms_graph={3,7:N0} ms_update={4,7:N0} vram={5,5}MiB" -f $exe,($dt/$n),($tot/$dt),($g/$n),($u/$n),($peak-$base))}
  else{
    $msg="";foreach($f in @("$dir.log","$dir.err")){if(-not $msg -and (Test-Path $f)){$msg=(Get-Content $f -EA SilentlyContinue|Select-String "FAIL|error"|Select-Object -First 1).Line}}
    $l=("exe={0,-18} FAIL {1} {2:N1}s" -f $exe,$msg,$dt)}
  Write-Output $l; Add-Content $OUT $l -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue }
Write-Output "BC_DONE"; Add-Content $OUT "BC_DONE" -Encoding utf8
