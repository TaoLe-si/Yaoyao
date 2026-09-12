# 干净基准：每个配置独立目录、独立日志，直接以日志中的 positions 求和 / 墙钟时间 = 真实吞吐。
# 同时记录显存峰值。用于在「装得下」与「够快」之间做选择。
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$X = $args[0]; if(-not $X){ $X = ".\build\train_shards_blas.exe" }
$OUT="D:\TaoVm\build\bench_clean.log"
Set-Content -Path $OUT -Value ("bench " + (Get-Date).ToString("HH:mm:ss") + "  exe=" + $X) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function Bench($tag,$cfg,$sl,$wd,$steps){
  $script:i++
  foreach($k in $cfg.Keys){ Set-Item "Env:\$k" $cfg[$k] }
  $dir="D:\TaoVm\build\bc_" + $script:i
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"$steps","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath $X -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(2400)
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
      if($ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ }
    }
  }
  if($n -ge 2){
    $line=("{0,-24} sl={1,3} wd={2,3} steps={3,3} time={4,8:N1}s positions={5,9} pos/s={6,8:N1} vram={7,5}MiB" -f $tag,$sl,$wd,$n,$dt,$tot,($tot/$dt),($peak-$base))
  } else {
    $msg=""; if(Test-Path "$dir.log"){$msg=(Get-Content "$dir.log" -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line}
    if(-not $msg){$msg=((Get-Content "$dir.out","$dir.err" -Tail 1 -EA SilentlyContinue)-join " ")}
    $line=("{0,-24} sl={1,3} wd={2,3} FAILED time={3:N1}s {4}" -f $tag,$sl,$wd,$dt,$msg)
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
$l2=@{TAO_CFG_LAYERS="2";TAO_CFG_D="1024";TAO_CFG_S="512";TAO_CFG_M="1024";TAO_CFG_DK="128"}
$c1536=@{TAO_CFG_LAYERS="12";TAO_CFG_D="1536";TAO_CFG_S="768";TAO_CFG_M="1536";TAO_CFG_DK="192"}
$c1280=@{TAO_CFG_LAYERS="18";TAO_CFG_D="1280";TAO_CFG_S="640";TAO_CFG_M="1280";TAO_CFG_DK="160"}
$c3072=@{TAO_CFG_LAYERS="2";TAO_CFG_D="3072";TAO_CFG_S="1536";TAO_CFG_M="3072";TAO_CFG_DK="384"}
Bench "L2d1024 基线" $l2 16 16 12
Bench "L2d1024 基线" $l2 32 16 12
Bench "L12d1536 (117M)" $c1536 8 8 6
Bench "L12d1536 (117M)" $c1536 8 4 6
Bench "L12d1536 (117M)" $c1536 16 4 6
Bench "L18d1280 (117M)" $c1280 8 8 6
Bench "L18d1280 (117M)" $c1280 16 4 6
Bench "L2d3072 (112M)" $c3072 8 8 6
Bench "L2d3072 (112M)" $c3072 16 4 6
Write-Output "BENCH_DONE"; Add-Content $OUT "BENCH_DONE" -Encoding utf8
