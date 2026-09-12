# 吞吐上限扫描：固定 23.6M 配置，只放大 slots（每步位置数 = slots*width），
# 判断训练是「被内核启动数限制」还是「被计算量限制」。开启 TAO_OPT_OFFLOAD（权重入内存）。
$ErrorActionPreference = "Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="2"; $env:TAO_CFG_D="1024"; $env:TAO_CFG_S="512"; $env:TAO_CFG_M="1024"; $env:TAO_CFG_DK="128"
$OUT = "D:\TaoVm\build\thru.log"
Set-Content -Path $OUT -Value ("thru " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
$base = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
foreach($off in @("1","0")){
  Set-Item "Env:\TAO_OPT_OFFLOAD" $off
  foreach($g in @(@(16,16),@(32,16),@(64,16),@(128,16),@(16,32),@(32,32),@(64,32))){
    $sl=$g[0]; $wd=$g[1]
    $dir="D:\TaoVm\build\tr_run"
    Remove-Item -Recurse -Force $dir -EA SilentlyContinue
    Remove-Item -Force "$dir.log","$dir.out","$dir.err" -EA SilentlyContinue
    $t0=Get-Date
    $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"10","$sl","$wd")
    $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
    $peak=0; $dl=(Get-Date).AddSeconds(240)
    while(-not $p.HasExited -and (Get-Date) -lt $dl){
      $m=[int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      if($m -gt $peak){$peak=$m}
      Start-Sleep -Milliseconds 250
    }
    if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
    $dt=((Get-Date)-$t0).TotalSeconds
    $upd=0
    if(Test-Path "$dir.log"){$upd=(Select-String -Path "$dir.log" -Pattern "^UPDATE " -EA SilentlyContinue|Measure-Object).Count}
    if($upd -ge 2){
      $line=("offload={0} slots={1,4} width={2,3} steps={3,2} perstep={4,7:N3}s {5,8:N1} pos/s vram={6,5} MiB" -f $off,$sl,$wd,$upd,($dt/$upd),(($upd*$sl*$wd)/$dt),($peak-$base))
    } else {
      $msg="";
      if(Test-Path "$dir.log"){$msg=(Get-Content "$dir.log" -EA SilentlyContinue|Select-String -Pattern "FAIL"|Select-Object -First 1).Line}
      $line=("offload={0} slots={1,4} width={2,3} FAILED {3:N1}s {4}" -f $off,$sl,$wd,$dt,$msg)
    }
    Write-Output $line; Add-Content $OUT $line -Encoding utf8
  }
}
Write-Output "THRU_DONE"; Add-Content $OUT "THRU_DONE" -Encoding utf8
