$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_D="1024"; $env:TAO_CFG_S="512"; $env:TAO_CFG_M="1024"; $env:TAO_CFG_DK="128"
$OUT="D:\TaoVm\build\depth.log"
Set-Content -Path $OUT -Value ("depth bisect " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($sl,$lay,$wd){
  $script:i++
  Set-Item "Env:\TAO_CFG_LAYERS" "$lay"
  $dir="D:\TaoVm\build\dp_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"3","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(600)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}
    Start-Sleep -Milliseconds 250
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0; $n=0
  if(Test-Path "$dir.log"){ foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){ if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ } } }
  if($n -ge 1){
    $line=("OK   L={0,3} wd={1,3} sl={2,3} depth={3,5} pos/步={4,6:N0} vram={5,5}MiB" -f $lay,$wd,$sl,($lay*$wd),($tot/$n),($peak-$base))
  } else {
    $msg=""; if(Test-Path "$dir.log"){$msg=(Get-Content "$dir.log" -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line}
    $short=if($msg -match "out of memory"){"OOM"}elseif($msg -match "not ready"){"NOTREADY"}else{"OTHER"}
    $line=("FAIL L={0,3} wd={1,3} sl={2,3} depth={3,5} {4} {5:N1}s" -f $lay,$wd,$sl,($lay*$wd),$short,$dt)
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
B 8 2 8 8
B 8 6 8 8
B 8 10 8 8
B 8 14 8 8
B 8 18 8 8
B 8 22 8 8
B 8 26 8 8
B 8 30 8 8
B 8 30 1 8
B 8 30 2 8
B 8 30 3 8
B 8 30 4 8
B 8 30 6 8
B 8 30 8 8
B 8 30 4 16
B 8 30 4 24
B 8 30 4 32
B 8 30 4 48
B 8 30 4 64
Write-Output "DEPTH_DONE"; Add-Content $OUT "DEPTH_DONE" -Encoding utf8
