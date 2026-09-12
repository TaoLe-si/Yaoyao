# 边际步成本探针：同一配置分别跑 4 / 12 / 24 步，用斜率分离「固定开销」与「每步计算」。
# 若斜率远小于平均步时，说明瓶颈是每步固定的优化器/投影开销，而非正反向计算 ——
# 那才是加宽真正慢的原因，也才是应该优化的地方。
$ErrorActionPreference = "Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_OPT_OFFLOAD="1"
$OUT="D:\TaoVm\build\marginal.log"
Set-Content -Path $OUT -Value ("marginal " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
function M($tag,$cfg,$sl,$wd,$steps){
  foreach($k in $cfg.Keys){ Set-Item "Env:\$k" $cfg[$k] }
  $dir="D:\TaoVm\build\mg_run"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log","$dir.out","$dir.err" -EA SilentlyContinue
  $t0=Get-Date
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"$steps","$sl","$wd")
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $dl=(Get-Date).AddSeconds(900)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){ Start-Sleep -Milliseconds 300 }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $upd=0; if(Test-Path "$dir.log"){$upd=(Select-String -Path "$dir.log" -Pattern "^UPDATE " -EA SilentlyContinue|Measure-Object).Count}
  $line=("{0,-22} slots={1,3} width={2,3} steps={3,3} actual={4,3} time={5,8:N2}s" -f $tag,$sl,$wd,$steps,$upd,$dt)
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  return @{t=$dt;n=$upd}
}
$cfgs=@(
  @{tag="L2 d1024 (23.6M)"; c=@{TAO_CFG_LAYERS="2";TAO_CFG_D="1024";TAO_CFG_S="512";TAO_CFG_M="1024";TAO_CFG_DK="128"}; sl=16; wd=16},
  @{tag="L12 d1536 (117M)"; c=@{TAO_CFG_LAYERS="12";TAO_CFG_D="1536";TAO_CFG_S="768";TAO_CFG_M="1536";TAO_CFG_DK="192"}; sl=8; wd=8},
  @{tag="L4 d2432 (117M)";  c=@{TAO_CFG_LAYERS="4";TAO_CFG_D="2432";TAO_CFG_S="1216";TAO_CFG_M="2432";TAO_CFG_DK="304"}; sl=8; wd=8},
  @{tag="L2 d3072 (112M)";  c=@{TAO_CFG_LAYERS="2";TAO_CFG_D="3072";TAO_CFG_S="1536";TAO_CFG_M="3072";TAO_CFG_DK="384"}; sl=8; wd=8}
)
foreach($x in $cfgs){
  foreach($k in @("TAO_CFG_LAYERS","TAO_CFG_D","TAO_CFG_S","TAO_CFG_M","TAO_CFG_DK")){ Remove-Item "Env:\$k" -EA SilentlyContinue }
  $r1=M $x.tag $x.c $x.sl $x.wd 4
  $r2=M $x.tag $x.c $x.sl $x.wd 20
  if($r1.n -ge 2 -and $r2.n -ge 2){
    $ds=$r2.n-$r1.n; $dtt=$r2.t-$r1.t
    if($ds -gt 0){
      $marg=$dtt/$ds; $fix=$r1.t-$marg*$r1.n
      $pos=$x.sl*$x.wd
      $line=("    -> 边际每步={0,7:N3}s  固定开销={1,7:N2}s  边际吞吐={2,7:N1} pos/s" -f $marg,$fix,($pos/$marg))
      Write-Output $line; Add-Content $OUT $line -Encoding utf8
    }
  }
}
Write-Output "MARGINAL_DONE"; Add-Content $OUT "MARGINAL_DONE" -Encoding utf8
