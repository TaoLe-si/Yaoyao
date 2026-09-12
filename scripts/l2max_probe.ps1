$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"
$LOG="D:\TaoVm\build\l2max_probe.log"
function Say($m){ Write-Output $m; Add-Content -Path $script:LOG -Value $m -Encoding utf8 }
Remove-Item -Force $LOG -ErrorAction SilentlyContinue
$configs = @(
  @{n="d768_s256_m768_dk96";    e=@{TAO_CFG_D="768";TAO_CFG_S="256";TAO_CFG_M="768";TAO_CFG_DK="96"}},
  @{n="d896_s384_m896_dk112";   e=@{TAO_CFG_D="896";TAO_CFG_S="384";TAO_CFG_M="896";TAO_CFG_DK="112"}},
  @{n="d1024_s384_m1024_dk128"; e=@{TAO_CFG_D="1024";TAO_CFG_S="384";TAO_CFG_M="1024";TAO_CFG_DK="128"}},
  @{n="d1024_s512_m1024_dk128"; e=@{TAO_CFG_D="1024";TAO_CFG_S="512";TAO_CFG_M="1024";TAO_CFG_DK="128"}},
  @{n="d1024_s768_m1024_dk256"; e=@{TAO_CFG_D="1024";TAO_CFG_S="768";TAO_CFG_M="1024";TAO_CFG_DK="256"}}
)
foreach($c in $configs){
  foreach($k in @("TAO_CFG_D","TAO_CFG_S","TAO_CFG_M","TAO_CFG_DK")){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  foreach($k in $c.e.Keys){ Set-Item "Env:\$k" $c.e[$k] }
  $dir="D:\TaoVm\build\l2m_"+$c.n
  Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue; Remove-Item -Force "$dir.log" -ErrorAction SilentlyContinue
  $t0=Get-Date
  & .\build\train_shards.exe data/probe_wiki build/tok_real_v1.bbp $dir 12 8 16 2>&1 | Out-Null
  $dt=((Get-Date)-$t0).TotalSeconds
  $u=0; if(Test-Path "$dir.log"){ $u=(Select-String -Path "$dir.log" -Pattern "^UPDATE " | Measure-Object).Count }
  if($u -gt 0){ Say ("{0,-26} 每步={1,5:N2}s {2,7:N0} 位置/s" -f $c.n,($dt/$u),(($u*8*8*16)/$dt)) }
  else { $err=""; if(Test-Path "$dir.log"){ $err=(Get-Content "$dir.log" | Select-Object -Last 2) -join " / " }; Say ("{0,-26} 失败: {1}" -f $c.n,$err) }
}
Say "L2MAX_DONE"
