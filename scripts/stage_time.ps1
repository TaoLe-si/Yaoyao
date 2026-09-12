# 阶段计时：分离 graph(fwd+bwd) 与 update(AdamW+project)
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"
$env:TAO_STAGE_TIMING="1"
$OUT="D:\TaoVm\build\stg.log"; Set-Content $OUT "stage timing" -Encoding utf8
foreach($cfg in @(@(1024,128,8,8),@(1024,128,16,16),@(2048,256,8,8),@(3200,400,8,8))){
  $d=$cfg[0];$dk=$cfg[1];$sl=$cfg[2];$wd=$cfg[3]
  Set-Item Env:\TAO_CFG_D $d; Set-Item Env:\TAO_CFG_S ([string]([int]$d/2))
  Set-Item Env:\TAO_CFG_M $d; Set-Item Env:\TAO_CFG_DK $dk
  $dir="D:\TaoVm\build\stg_$($d)_$($sl)"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"4","$sl","$wd")
  & .\build\train_shards.exe @ar 2>&1 | Out-Null
  if(Test-Path "$dir.log"){
    $upd=Get-Content "$dir.log" | Where-Object {$_ -match "^UPDATE "}
    $g=0.0;$u=0.0;$c=0
    foreach($ln in $upd){ if($ln -match "ms_graph=([0-9.]+) ms_upd=([0-9.]+) ms_project_sum=([0-9.]+) ms_total=([0-9.]+)"){$g+=[double]$matches[1];$u+=[double]$matches[2];$c++} }
    if($c){ $l2=("d={0,5} dk={1,4} sl={2,2} wd={3,2} 步={4} ms_graph={5,8:N0} ms_update={6,8:N0} 图占比={7,4:N0}% 更新占比={8,4:N0}%" -f $d,$dk,$sl,$wd,$c,($g/$c),($u/$c),(100*$g/($g+$u)),(100*$u/($g+$u))) }
    else{$l2="d=$d 无计时行"}
  } else {$l2="d=$d 无日志"}
  Write-Output $l2; Add-Content $OUT $l2 -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue }
Write-Output "STG_DONE"; Add-Content $OUT "STG_DONE" -Encoding utf8
