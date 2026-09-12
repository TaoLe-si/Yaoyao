# 诊断：主机制 vs 显存，d=1024 小配置，测量 GPU 利用率
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="2"; $env:TAO_CFG_D="1024"; $env:TAO_CFG_S="512"; $env:TAO_CFG_M="1024"; $env:TAO_CFG_DK="128"
$OUT="D:\TaoVm\build\hwdiag.log"; Set-Content $OUT "diag" -Encoding utf8
foreach($mode in @("1","0")){
  if($mode -eq "1"){Set-Item Env:\TAO_HOST_WEIGHTS "1"}else{Set-Item Env:\TAO_HOST_WEIGHTS "0"}
  $dir="D:\TaoVm\build\hd_$mode"; Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"5","16","16")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $sat=@(); $dl=(Get-Date).AddSeconds(600)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $u=(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits); $sat+=[int]$u
    Start-Sleep -Milliseconds 300 }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds; $tot=0;$n=0
  if(Test-Path "$dir.log"){foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){$tot+=[long]$matches[1];$n++}}}
  $avg=if($sat.Count){[math]::Round(($sat|Measure-Object -Average).Average,1)}else{0}
  $ln2=("HOST_WEIGHTS={0} 步数={1} t/步={2:N2}s tok/s={3:N0} GPU均值={4}%" -f $mode,$n,($(if($n){$dt/$n}else{0})),$(if($dt){$tot/$dt}else{0}),$avg)
  Write-Output $ln2; Add-Content $OUT $ln2 -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue }
Write-Output "DIAG_DONE"; Add-Content $OUT "DIAG_DONE" -Encoding utf8
