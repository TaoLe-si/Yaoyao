# d=3200 固定，扫 dk 与 slots/width：分离「激活显存」与「delta内存计算」的影响
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="2"; $env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"
Remove-Item Env:\TAO_OPT_OFFLOAD -EA SilentlyContinue
$OUT="D:\TaoVm\build\dk2.log"
Set-Content -Path $OUT -Value ("d3200 dk sweep " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($dk,$sl,$wd){
  $script:i++
  Set-Item "Env:\TAO_CFG_DK" "$dk"
  $dir="D:\TaoVm\build\k_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"6","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(1200)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}
    Start-Sleep -Milliseconds 250
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0; $n=0
  if(Test-Path "$dir.log"){ foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){ if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ } } }
  if($n -ge 1){
    $line=("OK   dk={0,3} sl={1,2} wd={2,2} t/步={3,6:N2}s tok/s={4,7:N0} ms/pos={5,7:N3} vram={6,5}MiB" -f $dk,$sl,$wd,($dt/$n),($tot/$dt),($dt/$tot*1000),($peak-$base))
  } else {
    $msg=""; foreach($f in @("$dir.log","$dir.err")){ if(-not $msg -and (Test-Path $f)){$msg=(Get-Content $f -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line} }
    $short=if($msg -match "out of memory"){"OOM"}elseif($msg -match "not ready"){"NOTREADY"}else{"OTHER"}
    $line=("FAIL dk={0,3} sl={1,2} wd={2,2} {3} {4:N1}s vram峰值={5,5}MiB" -f $dk,$sl,$wd,$short,$dt,($peak-$base))
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
B 400 8 8
B 256 8 8
B 128 8 8
B 256 8 16
B 256 16 8
B 128 16 8
B 128 16 16
B 256 16 16
Write-Output "DK2_DONE"; Add-Content $OUT "DK2_DONE" -Encoding utf8
