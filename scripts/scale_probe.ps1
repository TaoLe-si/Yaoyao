# 容量-吞吐探针：同批次下测不同配置的每步耗时，用于外推可训练容量上限。
$ErrorActionPreference = "Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"
$env:TAO_CPU_THREADS="4"
$OUT = "D:\TaoVm\build\scale_probe.log"
function Say($m){ $s = $m; Write-Output $s; Add-Content -Path $OUT -Value $s -Encoding utf8 }
Remove-Item -Force $OUT -ErrorAction SilentlyContinue
$configs = @(
  @{n="L2_d512_s128_m512";   e=@{}},
  @{n="L4_d512_s128_m512";   e=@{TAO_CFG_LAYERS="4"}},
  @{n="L8_d512_s128_m512";   e=@{TAO_CFG_LAYERS="8"}},
  @{n="L4_d768_s192_m768";   e=@{TAO_CFG_LAYERS="4";TAO_CFG_D="768";TAO_CFG_S="192";TAO_CFG_M="768";TAO_CFG_DK="96"}},
  @{n="L8_d512_s256_m1024";  e=@{TAO_CFG_LAYERS="8";TAO_CFG_S="256";TAO_CFG_M="1024";TAO_CFG_DK="128"}}
)
foreach($c in $configs){
  foreach($k in @("TAO_CFG_LAYERS","TAO_CFG_D","TAO_CFG_S","TAO_CFG_M","TAO_CFG_DK")){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  foreach($k in $c.e.Keys){ Set-Item "Env:\$k" $c.e[$k] }
  $out = "D:\TaoVm\build\sc_" + $c.n
  Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
  Remove-Item -Force "$out.log" -ErrorAction SilentlyContinue
  $t0 = Get-Date
  & .\build\train_shards.exe data/probe_wiki build/tok_real_v1.bbp $out 12 8 16 2>&1 | Out-Null
  $dt = ((Get-Date) - $t0).TotalSeconds
  $updates = 0
  if(Test-Path "$out.log"){ $updates = (Select-String -Path "$out.log" -Pattern "^UPDATE " | Measure-Object).Count }
  $mem = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if($updates -gt 0){ Say ("{0,-22} 步={1,3}  总用时={2,6:N1}s  每步={3,5:N2}s  {4,7:N2} 位置/s  GPU={5} MiB" -f $c.n,$updates,$dt,($dt/$updates),(($updates*8*8*16)/$dt),$mem) }
  else { Say ("{0,-22} 失败或未产出 UPDATE  (总用时 {1:N1}s)" -f $c.n,$dt) }
}
Say "SCALE_PROBE_DONE"
