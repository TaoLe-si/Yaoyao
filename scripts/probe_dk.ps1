# dk 杠杆验证：dk 减半 ⇒ 激活减半 ⇒ 可容纳更深的 117M 模型
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_D="1024"; $env:TAO_CFG_S="512"; $env:TAO_CFG_M="1024"
$OUT="D:\TaoVm\build\dk.log"
Set-Content -Path $OUT -Value ("dk sweep " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($lay,$dk,$sl,$wd){
  $script:i++
  Set-Item "Env:\TAO_CFG_LAYERS" "$lay"; Set-Item "Env:\TAO_CFG_DK" "$dk"
  $dir="D:\TaoVm\build\dk_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"5","$sl","$wd")
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
    $line=("OK   L={0,3} dk={1,3} sl={2,3} wd={3,3} steps={4} pos/步={5,6:N0} t/步={6,7:N2}s tok/s={7,8:N0} vram={8,5}MiB" -f $lay,$dk,$sl,$wd,$n,($tot/$n),($dt/$n),($tot/$dt),($peak-$base))
  } else {
    $msg=""; if(Test-Path "$dir.log"){$msg=(Get-Content "$dir.log" -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line}
    if(-not $msg -and (Test-Path "$dir.err")){$msg=(Get-Content "$dir.err" -EA SilentlyContinue|Select-String -Pattern "FAIL|require"|Select-Object -First 1).Line}
    $short=if($msg -match "out of memory"){"OOM"}elseif($msg -match "not ready"){"NOTREADY"}else{"OTHER "+$msg}
    $line=("FAIL L={0,3} dk={1,3} sl={2,3} wd={3,3} {4} {5:N1}s" -f $lay,$dk,$sl,$wd,$short,$dt)
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
B 30 96 8 8
B 30 64 8 8
B 30 96 8 16
B 30 64 8 16
B 30 64 16 8
B 30 96 6 8
B 30 80 8 8
B 30 64 16 16
Write-Output "DK_DONE"; Add-Content $OUT "DK_DONE" -Encoding utf8
