# 加宽配置的 (slots,width) 可行域扫描：激活显存 ≈ 参数量 × slots × width / 15 × 4B，
# 故必须同时缩小 slots 与 width 才可能装进 8GB。同时测吞吐以换算时间预算。
$ErrorActionPreference = "Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$OUT = "D:\TaoVm\build\wide_fit.log"
Set-Content -Path $OUT -Value ("fit sweep " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
$cases = @(
  @{n="L12 d1536 (117M)"; e=@{TAO_CFG_LAYERS="12";TAO_CFG_D="1536";TAO_CFG_S="768";TAO_CFG_M="1536";TAO_CFG_DK="192"}},
  @{n="L4 d2432 (117M)";  e=@{TAO_CFG_LAYERS="4";TAO_CFG_D="2432";TAO_CFG_S="1216";TAO_CFG_M="2432";TAO_CFG_DK="304"}},
  @{n="L2 d3072 (112M)";  e=@{TAO_CFG_LAYERS="2";TAO_CFG_D="3072";TAO_CFG_S="1536";TAO_CFG_M="3072";TAO_CFG_DK="384"}}
)
$grid = @(@(16,16),@(16,8),@(12,12),@(8,16),@(8,8),@(6,16))
foreach($c in $cases){
  foreach($k in $c.e.Keys){ Set-Item "Env:\$k" $c.e[$k] }
  foreach($g in $grid){
    $sl=$g[0]; $wd=$g[1]
    $dir = "D:\TaoVm\build\fw_run"
    Remove-Item -Recurse -Force $dir -EA SilentlyContinue
    Remove-Item -Force "$dir.log","$dir.out","$dir.err" -EA SilentlyContinue
    $t0 = Get-Date
    $ar = @("build\probe_one","build\tok_v2.bbp",$dir,"8","$sl","$wd")
    $p = Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
    $peak=0; $dl=(Get-Date).AddSeconds(200)
    while(-not $p.HasExited -and (Get-Date) -lt $dl){
      $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      if($m -gt $peak){$peak=$m}
      Start-Sleep -Milliseconds 250
    }
    if(-not $p.HasExited){ $p.Kill(); $p.WaitForExit() }
    $dt=((Get-Date)-$t0).TotalSeconds
    $upd=0
    if(Test-Path "$dir.log"){ $upd=(Select-String -Path "$dir.log" -Pattern "^UPDATE " -EA SilentlyContinue | Measure-Object).Count }
    if($upd -ge 2){
      $line=("{0,-20} slots={1,3} width={2,3} steps={3,2} {4,8:N1}s perstep={5,6:N3}s {6,8:N1} pos/s vram={7,5} MiB" -f $c.n,$sl,$wd,$upd,$dt,($dt/$upd),(($upd*$sl*$wd)/$dt),($peak-$base))
    } else {
      $msg="";
      if(Test-Path "$dir.log"){ $msg=(Get-Content "$dir.log" -EA SilentlyContinue | Select-String -Pattern "FAIL" | Select-Object -First 1).Line }
      if(-not $msg){ $msg=((Get-Content "$dir.out","$dir.err" -Tail 1 -EA SilentlyContinue) -join " ") }
      $line=("{0,-20} slots={1,3} width={2,3} FAILED {3:N1}s {4}" -f $c.n,$sl,$wd,$dt,$msg)
    }
    Write-Output $line; Add-Content $OUT $line -Encoding utf8
  }
}
Write-Output "WIDE_FIT_DONE"; Add-Content $OUT "WIDE_FIT_DONE" -Encoding utf8
