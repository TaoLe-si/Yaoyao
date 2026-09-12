# L=2 固定，扫 d。权重驻内存为默认（offload=1）。每个 d 试多个 slots 以分离 slots 与 d。
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="2"
Remove-Item Env:\TAO_OPT_OFFLOAD -EA SilentlyContinue
$OUT="D:\TaoVm\build\sd2.log"
Set-Content -Path $OUT -Value ("L2 sweep d, weights-in-RAM default " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($d,$s,$dk,$sl,$wd){
  $script:i++
  Set-Item "Env:\TAO_CFG_D" "$d"; Set-Item "Env:\TAO_CFG_S" "$s"; Set-Item "Env:\TAO_CFG_DK" "$dk"; Set-Item "Env:\TAO_CFG_M" "$d"
  $dir="D:\TaoVm\build\q_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"3","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(900)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}
    Start-Sleep -Milliseconds 250
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0; $n=0; $off="?"
  if(Test-Path "$dir.log"){ foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){
    if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ }
    if($ln -match "offload=(\d)"){ $off=$matches[1] } } }
  if($n -ge 1){
    $line=("OK   d={0,4} s={1,4} dk={2,3} sl={3,3} wd={4,3} off={5} pos/步={6,6:N0} t/步={7,6:N2}s tok/s={8,8:N0} vram={9,5}MiB" -f $d,$s,$dk,$sl,$wd,$off,($tot/$n),($dt/$n),($tot/$dt),($peak-$base))
  } else {
    $msg=""; foreach($f in @("$dir.log","$dir.err")){ if(-not $msg -and (Test-Path $f)){$msg=(Get-Content $f -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line} }
    $short=if($msg -match "out of memory"){"OOM"}elseif($msg -match "not ready"){"NOTREADY"}else{"OTHER: "+(($msg -split ":")[-1])}
    $line=("FAIL d={0,4} sl={1,3} wd={2,3} {3} {4:N1}s vram峰值={5,5}MiB" -f $d,$sl,$wd,$short,$dt,($peak-$base))
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
B 1024 512 128 16 16
B 2048 1024 256 8 8
B 2048 1024 256 16 16
B 3072 1536 384 8 8
B 3072 1536 384 16 16
B 3200 1600 400 8 8
B 3200 1600 400 16 16
B 2560 1280 320 16 16
Write-Output "SD2_DONE"; Add-Content $OUT "SD2_DONE" -Encoding utf8
