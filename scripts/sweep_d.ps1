# L=2 固定，扫 d：显存峰值 + 成败。权重默认驻内存（TAO_OPT_OFFLOAD 默认 1）
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="2"; $env:TAO_CFG_M=""
Remove-Item Env:\TAO_OPT_OFFLOAD -EA SilentlyContinue
$OUT="D:\TaoVm\build\sweep_d.log"
Set-Content -Path $OUT -Value ("L2 sweep d (weights in RAM) " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($d,$s,$dk){
  $script:i++
  Set-Item "Env:\TAO_CFG_D" "$d"; Set-Item "Env:\TAO_CFG_S" "$s"; Set-Item "Env:\TAO_CFG_DK" "$dk"; Set-Item "Env:\TAO_CFG_M" "$d"
  $dir="D:\TaoVm\build\sd_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"3","8","8")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(600)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}
    Start-Sleep -Milliseconds 200
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0; $n=0; $off="?"
  if(Test-Path "$dir.log"){ foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){
    if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ }
    if($ln -match "offload=(\d)"){ $off=$matches[1] } } }
  if($n -ge 1){
    $line=("OK   d={0,4} s={1,4} dk={2,3} offload={3} steps={4} pos/步={5,6:N0} t/步={6,6:N2}s tok/s={7,8:N0} vram={8,5}MiB" -f $d,$s,$dk,$off,$n,($tot/$n),($dt/$n),($tot/$dt),($peak-$base))
  } else {
    $msg=""; foreach($f in @("$dir.log","$dir.err")){ if(-not $msg -and (Test-Path $f)){$msg=(Get-Content $f -EA SilentlyContinue|Select-String -Pattern "FAIL|require"|Select-Object -First 1).Line} }
    $short=if($msg -match "out of memory"){"OOM"}elseif($msg -match "not ready"){"NOTREADY"}else{"OTHER "+(($msg -split ":")[-1])}
    $line=("FAIL d={0,4} s={1,4} dk={2,3} {3} {4:N1}s vram峰值={5,5}MiB" -f $d,$s,$dk,$short,$dt,($peak-$base))
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
B 1024 512 128
B 1280 640 160
B 1536 768 192
B 1792 896 224
B 2048 1024 256
B 2304 1152 288
B 2560 1280 320
B 2816 1408 352
B 3072 1536 384
B 3200 1600 400
Write-Output "SWEEPD_DONE"; Add-Content $OUT "SWEEPD_DONE" -Encoding utf8
