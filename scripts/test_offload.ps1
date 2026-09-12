# 只解析 UPDATE 行；开启 TAO_OPT_OFFLOAD 看加宽配置能否装下，并测真实吞吐。
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$OUT="D:\TaoVm\build\off_test.log"
Set-Content -Path $OUT -Value ("off test " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function Bench($off,$tag,$cfg,$sl,$wd,$steps){
  $script:i++
  Set-Item "Env:\TAO_OPT_OFFLOAD" $off
  foreach($k in $cfg.Keys){ Set-Item "Env:\$k" $cfg[$k] }
  $dir="D:\TaoVm\build\ot_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"$steps","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards_blas.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(3600)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if($m -gt $peak){$peak=$m}
    Start-Sleep -Milliseconds 400
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0; $n=0
  if(Test-Path "$dir.log"){
    foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){
      if($ln -match "^UPDATE "){ if($ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ } }
    }
  }
  if($n -ge 2){
    $line=("{0} {1,-20} sl={2,3} wd={3,3} steps={4,3} time={5,8:N1}s positions={6,8} pos/s={7,9:N0} vram={8,5}MiB" -f $off,$tag,$sl,$wd,$n,$dt,$tot,($tot/$dt),($peak-$base))
  } else {
    $msg=""; if(Test-Path "$dir.log"){$msg=(Get-Content "$dir.log" -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line}
    $line=("{0} {1,-20} sl={2,3} wd={3,3} FAILED time={4:N1}s {5}" -f $off,$tag,$sl,$wd,$dt,$msg)
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
}
$l2=@{TAO_CFG_LAYERS="2";TAO_CFG_D="1024";TAO_CFG_S="512";TAO_CFG_M="1024";TAO_CFG_DK="128"}
$c1536=@{TAO_CFG_LAYERS="12";TAO_CFG_D="1536";TAO_CFG_S="768";TAO_CFG_M="1536";TAO_CFG_DK="192"}
$c1280=@{TAO_CFG_LAYERS="18";TAO_CFG_D="1280";TAO_CFG_S="640";TAO_CFG_M="1280";TAO_CFG_DK="160"}
$c3072=@{TAO_CFG_LAYERS="2";TAO_CFG_D="3072";TAO_CFG_S="1536";TAO_CFG_M="3072";TAO_CFG_DK="384"}
Bench 1 "L2d1024 基线" $l2 32 16 10
Bench 1 "L12d1536" $c1536 8 8 4
Bench 1 "L12d1536" $c1536 16 8 4
Bench 1 "L12d1536" $c1536 16 16 4
Bench 1 "L18d1280" $c1280 16 8 4
Bench 1 "L18d1280" $c1280 16 16 4
Bench 1 "L2d3072" $c3072 8 8 4
Bench 1 "L2d3072" $c3072 16 8 4
Write-Output "OFF_TEST_DONE"; Add-Content $OUT "OFF_TEST_DONE" -Encoding utf8
