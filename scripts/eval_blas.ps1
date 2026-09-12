# cuBLAS 评估：1) 数值等价性（同配置同种子，逐行比对 UPDATE 的 NLL）
#            2) 吞吐（不同宽度下的边际每步与 pos/s）
$ErrorActionPreference = "Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$OUT="D:\TaoVm\build\blas_eval.log"
Set-Content -Path $OUT -Value ("blas eval " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
function RunOne($exe,$cfg,$sl,$wd,$steps,$tag){
  foreach($k in $cfg.Keys){ Set-Item "Env:\$k" $cfg[$k] }
  $dir="D:\TaoVm\build\be_" + $tag
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"$steps","$sl","$wd")
  $p=Start-Process -FilePath $exe -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $dl=(Get-Date).AddSeconds(1500)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){ Start-Sleep -Milliseconds 300 }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  return "$dir.log"
}
function Nlls($log){
  if(-not (Test-Path $log)){return @()}
  return (Get-Content $log -EA SilentlyContinue | Select-String -Pattern "^UPDATE " | ForEach-Object {
    if($_.Line -match "train_preupdate_NLL=([0-9.]+)"){ [double]$matches[1] } })
}
$l2=@{TAO_CFG_LAYERS="2";TAO_CFG_D="1024";TAO_CFG_S="512";TAO_CFG_M="1024";TAO_CFG_DK="128"}
$wide=@(
  @{n="L12 d1536 (117M)"; c=@{TAO_CFG_LAYERS="12";TAO_CFG_D="1536";TAO_CFG_S="768";TAO_CFG_M="1536";TAO_CFG_DK="192"}},
  @{n="L4 d2432 (117M)";  c=@{TAO_CFG_LAYERS="4";TAO_CFG_D="2432";TAO_CFG_S="1216";TAO_CFG_M="2432";TAO_CFG_DK="304"}},
  @{n="L2 d3072 (112M)";  c=@{TAO_CFG_LAYERS="2";TAO_CFG_D="3072";TAO_CFG_S="1536";TAO_CFG_M="3072";TAO_CFG_DK="384"}}
)
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
Write-Output "===== A. 数值等价性（L2 d1024, slots=16 width=16, 3 步）====="
$a=RunOne ".\build\train_shards.exe"      $l2 16 16 3 "eq_ref"
$b=RunOne ".\build\train_shards_blas.exe" $l2 16 16 3 "eq_blas"
$na=Nlls $a; $nb=Nlls $b
Write-Output ("  非BLAS NLL: " + ($na -join ", "))
Write-Output (""  + "  BLAS   NLL: " + ($nb -join ", "))
if($na.Count -gt 0 -and $nb.Count -gt 0){
  $mx=0; $cnt=[Math]::Min($na.Count,$nb.Count)
  for($i=0;$i -lt $cnt;$i++){ $d=[Math]::Abs($na[$i]-$nb[$i]); if($d -gt $mx){$mx=$d} }
  $line=("  前 {0} 步最大 NLL 差 = {1:E3}" -f $cnt,$mx)
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
} else { Write-Output "  !! 无法比对"; Add-Content $OUT "eq compare failed" -Encoding utf8 }
Write-Output ""
Write-Output "===== B. 吞吐（BLAS，边际每步）====="
Add-Content $OUT ("--- BLAS throughput @ " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
function Thru($tag,$cfg,$sl,$wd){
  $t0=Get-Date
  $lg=RunOne ".\build\train_shards_blas.exe" $cfg $sl $wd 16 "th_$tag"
  $dt=((Get-Date)-$t0).TotalSeconds
  $n=(Nlls $lg).Count
  if($n -ge 2){
    $line=("{0,-20} slots={1,3} width={2,3} steps={3,2} time={4,8:N2}s perstep={5,7:N3}s {6,8:N1} pos/s" -f $tag,$sl,$wd,$n,$dt,($dt/$n),(($n*$sl*$wd)/$dt))
  } else {
    $msg=""; if(Test-Path $lg){ $msg=(Get-Content $lg | Select-String -Pattern "FAIL" | Select-Object -First 1).Line }
    $line=("{0,-20} slots={1,3} width={2,3} FAILED {3} {4}" -f $tag,$sl,$wd,[math]::Round($dt,1),$msg)
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
}
Thru "L2d1024" $l2 16 16
Thru "L2d1024" $l2 32 16
Thru "L2d1024" $l2 32 32
foreach($w in $wide){ Thru $w.n $w.c 8 8; Thru $w.n $w.c 8 16; Thru $w.n $w.c 16 8; Thru $w.n $w.c 16 16 }
Write-Output "BLAS_EVAL_DONE"; Add-Content $OUT "BLAS_EVAL_DONE" -Encoding utf8
