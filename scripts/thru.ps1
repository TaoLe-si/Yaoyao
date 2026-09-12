# 用 20 步/分片把 SHARD_LOADED 摊薄，测真实吞吐（BLAS 与非BLAS）
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"
$env:TAO_STAGE_TIMING="1"
$env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"; $env:TAO_CFG_DK="400"
$OUT="D:\TaoVm\build\thru.log"; Set-Content $OUT "throughput 20 steps/shard" -Encoding utf8
foreach($exe in @("train_shards_blas","train_shards")){
  $dir="D:\TaoVm\build\th_$exe"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"20","8","8")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\$exe.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $dl=(Get-Date).AddSeconds(900)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){Start-Sleep -Milliseconds 300}
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds;$tot=0;$n=0;$g=0.0;$u=0.0
  if(Test-Path "$dir.log"){foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){
    if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){$tot+=[long]$matches[1];$n++}
    if($ln -match "ms_graph=([0-9.]+) ms_upd=([0-9.]+)"){$g+=[double]$matches[1];$u+=[double]$matches[2]} }}
  if($n -ge 1){
    $l=("exe={0,-18} 步={1,3} 总={2,6:N1}s t/步={3,5:N2}s tok/s(含加载)={4,7:N0} ms_graph={5,7:N0} ms_update={6,7:N0} 引擎t/步={7,5:N2}s tok/s(纯引擎)={8,7:N0}" -f $exe,$n,$dt,($dt/$n),($tot/$dt),($g/$n),($u/$n),((($g+$u)/1000)),($tot/(($g+$u)/1000)))
  } else {$l="exe=$exe FAIL"}
  Write-Output $l; Add-Content $OUT $l -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue }
Write-Output "THRU_DONE"; Add-Content $OUT "THRU_DONE" -Encoding utf8
