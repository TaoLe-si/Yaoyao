# L=30 d=1024 (119M) 用「多 slots × 小 width」压激活显存
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="30"; $env:TAO_CFG_D="1024"; $env:TAO_CFG_S="512"; $env:TAO_CFG_M="1024"; $env:TAO_CFG_DK="128"
$OUT="D:\TaoVm\build\l30.log"
Set-Content -Path $OUT -Value ("L30 narrow-width " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($sl,$wd){
  $script:i++
  $dir="D:\TaoVm\build\n30_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"5","$sl","$wd")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(900)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}
    Start-Sleep -Milliseconds 250
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0; $n=0
  if(Test-Path "$dir.log"){ foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){ if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){ $tot += [long]$matches[1]; $n++ } } }
  if($n -ge 1){
    $line=("OK   sl={0,3} wd={1,2} depth={2,4} steps={3} pos/步={4,6:N0} t/步={5,7:N2}s tok/s={6,8:N0} vram={7,5}MiB" -f $sl,$wd,(30*$wd),$n,($tot/$n),($dt/$n),($tot/$dt),($peak-$base))
  } else {
    $msg=""; if(Test-Path "$dir.log"){$msg=(Get-Content "$dir.log" -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line}
    $short=if($msg -match "out of memory"){"OOM"}elseif($msg -match "not ready"){"NOTREADY"}else{"OTHER: $msg"}
    $line=("FAIL sl={0,3} wd={1,2} depth={2,4} {3} {4:N1}s" -f $sl,$wd,(30*$wd),$short,$dt)
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
B 32 1
B 16 2
B 8 4
B 32 2
B 16 4
B 64 1
B 64 2
B 128 1
Write-Output "L30_DONE"; Add-Content $OUT "L30_DONE" -Encoding utf8
