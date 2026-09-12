# 分解每步耗时：(a) 现版 (b) GPU 健康规约版 —— 分离「CPU 串行扫描」与「PCIe 来回搬运」
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
Remove-Item Env:\TAO_OPT_OFFLOAD -EA SilentlyContinue
$OUT="D:\TaoVm\build\bd.log"
Set-Content -Path $OUT -Value ("breakdown " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($exe,$tag,$d,$s,$dk,$sl,$wd,$steps){
  $script:i++
  Set-Item "Env:\TAO_CFG_LAYERS" "2"; Set-Item "Env:\TAO_CFG_D" "$d"; Set-Item "Env:\TAO_CFG_S" "$s"; Set-Item "Env:\TAO_CFG_M" "$d"; Set-Item "Env:\TAO_CFG_DK" "$dk"
  $dir="D:\TaoVm\build\bd_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"$steps","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath $exe -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $u=@(); $peak=0; $dl=(Get-Date).AddSeconds(1500)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $r=(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) -split ","
    $u += [int]$r[0].Trim(); $v=[int]$r[1].Trim(); if($v -gt $peak){$peak=$v}
    Start-Sleep -Milliseconds 250
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0; $n=0
  if(Test-Path "$dir.log"){ foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){ if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ } } }
  $gpu=if($u.Count){[math]::Round(($u|Measure-Object -Average).Average,1)}else{0}
  if($n -ge 1){
    $line=("{0,-22} d={1,4} sl={2,2} wd={3,2} steps={4,2} t/步={5,6:N2}s tok/s={6,7:N0} GPU均值={7,5}% vram峰值={8,5}MiB" -f $tag,$d,$sl,$wd,$n,($dt/$n),($tot/$dt),$gpu,$peak)
  } else {
    $msg=""; foreach($f in @("$dir.log","$dir.err")){ if(-not $msg -and (Test-Path $f)){$msg=(Get-Content $f -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line} }
    $line=("{0,-22} d={1,4} sl={2,2} wd={3,2} FAILED {4:N1}s {5}" -f $tag,$d,$sl,$wd,$dt,$msg)
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
B ".\build\train_shards.exe"    "现版(CPU扫描)" 1024 512 128 16 16 10
if(Test-Path ".\build\train_shards_gh.exe"){ B ".\build\train_shards_gh.exe" "GPU规约版" 1024 512 128 16 16 10 }
B ".\build\train_shards.exe"    "现版(CPU扫描)" 3200 1600 400 8 8 6
if(Test-Path ".\build\train_shards_gh.exe"){ B ".\build\train_shards_gh.exe" "GPU规约版" 3200 1600 400 8 8 6 }
Write-Output "BD_DONE"; Add-Content $OUT "BD_DONE" -Encoding utf8
