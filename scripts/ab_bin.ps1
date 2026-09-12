# A/B: train_shards.exe (unified cuBLAS) vs train_shards_blas.exe (pre-unification)
# 交替 3 轮以消除热漂移/外部显存占用导致的时间漂移。
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_STAGE_TIMING="1"; $env:TAO_CFG_LAYERS="2"
$env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"; $env:TAO_CFG_DK="400"
$OUT="D:\TaoVm\build\ab_bin.log"; Set-Content $OUT "A/B binary comparison d=3200 sl8 wd8" -Encoding utf8
foreach($round in 1..3){
  foreach($exe in @("train_shards.exe","train_shards_blas.exe")){
    Remove-Item -Recurse -Force build\ab_tmp -EA SilentlyContinue; Remove-Item -Force build\ab_tmp.log -EA SilentlyContinue
    $t0=Get-Date
    $p=Start-Process -FilePath ".\build\$exe" -ArgumentList @("build\probe_one","build\tok_v2.bbp","build\ab_tmp","8","8","8") -PassThru -NoNewWindow
    $peak=0; while(-not $p.HasExited){$m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}; Start-Sleep -Milliseconds 150}
    $dt=((Get-Date)-$t0).TotalSeconds
    $g=@(); $u=@(); $pr=@()
    if(Test-Path build\ab_tmp.log){foreach($ln in (Get-Content build\ab_tmp.log)){
      if($ln -match "^UPDATE"){ if($ln -match "ms_graph=([0-9.]+)"){$g+=[double]$matches[1]}; if($ln -match "ms_upd=([0-9.]+)"){$u+=[double]$matches[1]}; if($ln -match "ms_project=([0-9.]+)"){$pr+=[double]$matches[1]} } }}
    $n=$g.Count
    if($n -ge 4){ $g2=($g | Select-Object -Skip 1 | Measure-Object -Average).Average; $u2=($u | Select-Object -Skip 1 | Measure-Object -Average).Average; $p2=($pr | Select-Object -Skip 1 | Measure-Object -Average).Average }
    else { $g2=0;$u2=0;$p2=0 }
    $l=("round={0} {1,-24} wall={2,6:N2}s peak={3,5}MiB ms_graph={4,7:N0} ms_upd={5,7:N0} ms_project={6,7:N0}" -f $round,$exe,$dt,$peak,$g2,$u2,$p2)
    Write-Output $l; Add-Content $OUT $l -Encoding utf8 } }
Write-Output "AB_BIN_DONE"; Add-Content $OUT "AB_BIN_DONE" -Encoding utf8
