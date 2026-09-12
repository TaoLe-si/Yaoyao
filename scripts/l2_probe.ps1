# 纯 L2 加宽探针：层数固定 2，只放大 d/s/m/dk，测吞吐与显存。
$ErrorActionPreference = "Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"
$env:TAO_CPU_THREADS="4"
$LOG = "D:\TaoVm\build\l2_probe.log"
function Say($m){ Write-Output $m; Add-Content -Path $script:LOG -Value $m -Encoding utf8 }
Remove-Item -Force $LOG -ErrorAction SilentlyContinue
$configs = @(
  @{n="d512_s128_m512_dk64";    e=@{}},
  @{n="d640_s192_m640_dk80";    e=@{TAO_CFG_D="640";TAO_CFG_S="192";TAO_CFG_M="640";TAO_CFG_DK="80"}},
  @{n="d768_s256_m768_dk96";    e=@{TAO_CFG_D="768";TAO_CFG_S="256";TAO_CFG_M="768";TAO_CFG_DK="96"}},
  @{n="d768_s256_m1024_dk128";  e=@{TAO_CFG_D="768";TAO_CFG_S="256";TAO_CFG_M="1024";TAO_CFG_DK="128"}},
  @{n="d1024_s384_m1536_dk192"; e=@{TAO_CFG_D="1024";TAO_CFG_S="384";TAO_CFG_M="1536";TAO_CFG_DK="192"}},
  @{n="d1024_s512_m2048_dk256"; e=@{TAO_CFG_D="1024";TAO_CFG_S="512";TAO_CFG_M="2048";TAO_CFG_DK="256"}}
)
foreach($c in $configs){
  foreach($k in @("TAO_CFG_LAYERS","TAO_CFG_D","TAO_CFG_S","TAO_CFG_M","TAO_CFG_DK")){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  Set-Item "Env:\TAO_CFG_LAYERS" "2"
  foreach($k in $c.e.Keys){ Set-Item "Env:\$k" $c.e[$k] }
  $dir = "D:\TaoVm\build\l2p_" + $c.n
  Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
  Remove-Item -Force "$dir.log" -ErrorAction SilentlyContinue
  $t0 = Get-Date
  & .\build\train_shards.exe data/probe_wiki build/tok_real_v1.bbp $dir 12 8 16 2>&1 | Out-Null
  $dt = ((Get-Date) - $t0).TotalSeconds
  $u = 0; if(Test-Path "$dir.log"){ $u = (Select-String -Path "$dir.log" -Pattern "^UPDATE " | Measure-Object).Count }
  $mem = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if($u -gt 0){ Say ("{0,-24} 步={1,3} 每步={2,5:N2}s {3,7:N0} 位置/s GPU={4} MiB" -f $c.n,$u,($dt/$u),(($u*8*8*16)/$dt),$mem) }
  else { Say ("{0,-24} 失败（{1:N1}s）" -f $c.n,$dt) }
}
Say "L2_PROBE_DONE"
