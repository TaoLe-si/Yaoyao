# slots/width 扫描：找出能跑通的最大并行度，并测吞吐与显存。
# 结论直接决定 117M 模型的时间预算。
$ErrorActionPreference = "Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$OUT = "D:\TaoVm\build\slots_sweep.log"
Set-Content -Path $OUT -Value ("scan " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$SLOTS = @(16,32,48,64,96,128,192,256)
$WIDTHS = @(16)
$cfg = "L" + $env:TAO_CFG_LAYERS + " d" + $env:TAO_CFG_D + " s" + $env:TAO_CFG_S + " m" + $env:TAO_CFG_M + " dk" + $env:TAO_CFG_DK
Write-Output ("CONFIG " + $cfg); Add-Content $OUT ("CONFIG " + $cfg) -Encoding utf8
$base = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
foreach($sl in $SLOTS){
  foreach($wd in $WIDTHS){
    $dir = "D:\TaoVm\build\sw_run"
    Remove-Item -Recurse -Force $dir -EA SilentlyContinue
    Remove-Item -Force "$dir.log","$dir.out","$dir.err" -EA SilentlyContinue
    $t0 = Get-Date
    $args_ = @("build\probe_one","build\tok_v2.bbp",$dir,"6","$sl","$wd")
    $p = Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $args_ -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
    $peak = 0; $deadline = (Get-Date).AddSeconds(180)
    while(-not $p.HasExited -and (Get-Date) -lt $deadline){
      $m = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      if($m -gt $peak){ $peak = $m }
      Start-Sleep -Milliseconds 300
    }
    if(-not $p.HasExited){ $p.Kill(); $p.WaitForExit() }
    $dt = ((Get-Date) - $t0).TotalSeconds
    $upd = 0
    if(Test-Path "$dir.log"){ $upd = (Select-String -Path "$dir.log" -Pattern "^UPDATE " -EA SilentlyContinue | Measure-Object).Count }
    if($upd -ge 2){
      $line = ("slots={0,4} width={1,3}  steps={2,2} time={3,7:N1}s perstep={4,7:N3}s {5,9:N1} pos/s  vram={6,5} MiB" -f $sl,$wd,$upd,$dt,($dt/$upd),(($upd*$sl*$wd)/$dt),($peak-$base))
    } else {
      $msg = ""
      if(Test-Path "$dir.log"){ $msg = (Get-Content "$dir.log" -EA SilentlyContinue | Select-String -Pattern "FAIL|error|out of memory" | Select-Object -First 1).Line }
      if(-not $msg){ $msg = ((Get-Content "$dir.out","$dir.err" -Tail 1 -EA SilentlyContinue) -join " ") }
      $line = ("slots={0,4} width={1,3}  FAILED time={2:N1}s  {3}" -f $sl,$wd,$dt,$msg)
    }
    Write-Output $line; Add-Content $OUT $line -Encoding utf8
  }
}
Write-Output "SLOTS_SWEEP_DONE"; Add-Content $OUT "SLOTS_SWEEP_DONE" -Encoding utf8
