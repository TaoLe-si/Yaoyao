# 权重驻内存版：d=3200 dk=400（不减 dk），扫描 slots/width 找吞吐上限
$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="2"; $env:TAO_CFG_D="3200"; $env:TAO_CFG_S="1600"; $env:TAO_CFG_M="3200"; $env:TAO_CFG_DK="400"
Remove-Item Env:\TAO_OPT_OFFLOAD -EA SilentlyContinue
$OUT="D:\TaoVm\build\hw_sweep.log"
Set-Content $OUT ("host-weighted sweep " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$i=0
function B($sl,$wd,$seg){
  $script:i++
  $dir="D:\TaoVm\build\w_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"6","$sl","$wd")
  if($seg -gt 0){$env:TAO_SEG="1"}else{Remove-Item Env:\TAO_SEG -EA SilentlyContinue}
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak=0; $dl=(Get-Date).AddSeconds(900)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){
    $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if($m -gt $peak){$peak=$m}
    Start-Sleep -Milliseconds 200
  }
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds
  $tot=0;$n=0
  if(Test-Path "$dir.log"){ foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){ if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){$tot+=[long]$matches[1];$n++} } }
  if($n -ge 1){
    $nll="-"; if((Get-Content "$dir.log" -EA SilentlyContinue | Select-String "train_preupdate_NLL=([0-9.]+)" | Select-Object -Last 1)){$nll=$matches[1]}
    $line=("OK   sl={0,2} wd={1,2} t/步={2,5:N2}s tok/s={3,7:N0} ms/pos={4,7:N3} vram={5,5}MiB NLL={6}" -f $sl,$wd,($dt/$n),($tot/$dt),($dt/$tot*1000),($peak-$base),$nll)
  }else{
    $msg=""; foreach($f in @("$dir.log","$dir.err")){ if(-not $msg -and (Test-Path $f)){$msg=(Get-Content $f -EA SilentlyContinue|Select-String "FAIL"|Select-Object -First 1).Line} }
    $short=if($msg -match "out of memory"){"OOM"}elseif($msg -match "not ready"){"NOTREADY"}else{"OTHER"}
    $line=("FAIL sl={0,2} wd={1,2} {2} {3:N1}s vram峰值={4,5}MiB" -f $sl,$wd,$short,$dt,($peak-$base))
  }
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
}
B  8  8 0
B 12  8 0
B 16  8 0
B  8 12 0
B  8 16 0
B 16 16 0
B 24  8 0
B 12 16 0
Write-Output "HW_DONE"; Add-Content $OUT "HW_DONE" -Encoding utf8
