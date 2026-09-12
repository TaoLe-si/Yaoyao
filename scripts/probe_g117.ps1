# A. L=30 d=1024 (119M) 可行性 + 吞吐（BLAS 与非 BLAS 对比）
# B. d=1536 非 BLAS，隔离 device not ready 是否 BLAS/图捕获特有
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$OUT="D:\TaoVm\build\g117.log"
Set-Content -Path $OUT -Value ("g117 " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($exe,$tag,$cfg,$sl,$wd,$steps){
  $script:i++
  foreach($k in $cfg.Keys){ Set-Item "Env:\$k" $cfg[$k] }
  $dir="D:\TaoVm\build\g_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"$steps","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath $exe -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(2400)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if($m -gt $peak){$peak=$m}
    Start-Sleep -Milliseconds 300
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0; $n=0
  if(Test-Path "$dir.log"){
    foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){
      if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ }
    }
  }
  if($n -ge 2){
    $line=("{0,-12} {1,-16} sl={2,3} wd={3,3} steps={4,3} time={5,8:N1}s pos/步={6,6:N0} pos/s={7,9:N0} vram={8,5}MiB" -f (Split-Path $exe -Leaf),$tag,$sl,$wd,$n,$dt,($tot/$n),($tot/$dt),($peak-$base))
  } else {
    $msg=""; if(Test-Path "$dir.log"){$msg=(Get-Content "$dir.log" -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line}
    $line=("{0,-12} {1,-16} sl={2,3} wd={3,3} FAILED {4:N1}s {5}" -f (Split-Path $exe -Leaf),$tag,$sl,$wd,$dt,$msg)
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
$L30=@{TAO_CFG_LAYERS="30";TAO_CFG_D="1024";TAO_CFG_S="512";TAO_CFG_M="1024";TAO_CFG_DK="128"}
$L29=@{TAO_CFG_LAYERS="29";TAO_CFG_D="1024";TAO_CFG_S="512";TAO_CFG_M="1024";TAO_CFG_DK="128"}
$d1536=@{TAO_CFG_LAYERS="12";TAO_CFG_D="1536";TAO_CFG_S="768";TAO_CFG_M="1536";TAO_CFG_DK="192"}
B ".\build\train_shards_blas.exe" "L30d1024 BLAS" $L30 8 8 8
B ".\build\train_shards_blas.exe" "L30d1024 BLAS" $L30 16 8 8
B ".\build\train_shards_blas.exe" "L30d1024 BLAS" $L30 16 16 6
B ".\build\train_shards.exe"      "L30d1024 原版" $L30 8 8 8
B ".\build\train_shards.exe"      "L30d1024 原版" $L30 16 16 6
B ".\build\train_shards_blas.exe" "L29d1024 BLAS" $L29 16 16 6
B ".\build\train_shards.exe"      "d1536 原版" $d1536 8 8 4
B ".\build\train_shards.exe"      "d1536 原版" $d1536 16 16 4
Write-Output "G117_DONE"; Add-Content $OUT "G117_DONE" -Encoding utf8
