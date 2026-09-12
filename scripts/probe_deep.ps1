# offload 默认开启（权重驻内存）后，d=1024 dk=128 能装多深
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_D="1024"; $env:TAO_CFG_S="512"; $env:TAO_CFG_M="1024"; $env:TAO_CFG_DK="128"
Remove-Item Env:\TAO_OPT_OFFLOAD -EA SilentlyContinue
$OUT="D:\TaoVm\build\deep.log"
Set-Content -Path $OUT -Value ("depth with weights-in-ram (default) " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($lay,$sl,$wd){
  $script:i++
  Set-Item "Env:\TAO_CFG_LAYERS" "$lay"
  $dir="D:\TaoVm\build\dw_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"4","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(1500)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}
    Start-Sleep -Milliseconds 300
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0; $n=0
  if(Test-Path "$dir.log"){ foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){ if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ } } }
  if($n -ge 1){
    $line=("OK   L={0,3} sl={1,3} wd={2,3} steps={3} pos/步={4,6:N0} t/步={5,7:N2}s tok/s={6,8:N0} vram={7,5}MiB" -f $lay,$sl,$wd,$n,($tot/$n),($dt/$n),($tot/$dt),($peak-$base))
  } else {
    $msg=""; if(Test-Path "$dir.log"){$msg=(Get-Content "$dir.log" -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line}
    $short=if($msg -match "out of memory"){"OOM"}elseif($msg -match "not ready"){"NOTREADY"}else{"OTHER"}
    $line=("FAIL L={0,3} sl={1,3} wd={2,3} {3} {4:N1}s" -f $lay,$sl,$wd,$short,$dt)
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
B 18 8 8
B 22 8 8
B 26 8 8
B 30 8 8
B 32 8 8
B 36 8 8
B 40 8 8
B 30 16 8
B 30 8 16
B 24 16 16
Write-Output "DEEP_DONE"; Add-Content $OUT "DEEP_DONE" -Encoding utf8
