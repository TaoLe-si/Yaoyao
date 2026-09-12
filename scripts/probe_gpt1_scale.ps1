# 117M 选型探针：单分片隔离纯训练吞吐，采样峰值显存。
$ErrorActionPreference = "Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
Remove-Item Env:\TAO_FRESH -EA SilentlyContinue
$OUT = "D:\TaoVm\build\gpt1_scale.log"
Set-Content -Path $OUT -Value ("探针启动 " + (Get-Date).ToString("HH:mm:ss")) -Encoding utf8
Write-Output ("探针启动 " + (Get-Date).ToString("HH:mm:ss"))

# 单分片工作目录：只放一片，隔离掉分片加载的干扰
$SRC = "D:\TaoVm\data\p1_wiki\shard_00000.bin"
$ONE = "D:\TaoVm\build\probe_one"
Remove-Item -Recurse -Force $ONE -EA SilentlyContinue
New-Item -ItemType Directory -Force -Path $ONE | Out-Null
Copy-Item $SRC (Join-Path $ONE "shard_00000.bin") -Force

$STEPS = 40
$cfg = @(
  @{n="对照 L2 d1024 s512 (23.6M)"; e=@{TAO_CFG_LAYERS="2";TAO_CFG_D="1024";TAO_CFG_S="512";TAO_CFG_M="1024";TAO_CFG_DK="128"}},
  @{n="L12 d1536 s768 (117.3M)"; e=@{TAO_CFG_LAYERS="12";TAO_CFG_D="1536";TAO_CFG_S="768";TAO_CFG_M="1536";TAO_CFG_DK="192"}},
  @{n="L18 d1280 s640 (116.9M)"; e=@{TAO_CFG_LAYERS="18";TAO_CFG_D="1280";TAO_CFG_S="640";TAO_CFG_M="1280";TAO_CFG_DK="160"}},
  @{n="L10 d1664 s832 (117.3M)"; e=@{TAO_CFG_LAYERS="10";TAO_CFG_D="1664";TAO_CFG_S="832";TAO_CFG_M="1664";TAO_CFG_DK="208"}},
  @{n="L4 d2432 s1216 (116.8M)"; e=@{TAO_CFG_LAYERS="4";TAO_CFG_D="2432";TAO_CFG_S="1216";TAO_CFG_M="2432";TAO_CFG_DK="304"}},
  @{n="L2 d3072 s1536 (111.7M)"; e=@{TAO_CFG_LAYERS="2";TAO_CFG_D="3072";TAO_CFG_S="1536";TAO_CFG_M="3072";TAO_CFG_DK="384"}}
)
$base = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
foreach($c in $cfg){
  foreach($k in @("TAO_CFG_LAYERS","TAO_CFG_D","TAO_CFG_S","TAO_CFG_M","TAO_CFG_DK")){ Remove-Item "Env:\$k" -EA SilentlyContinue }
  foreach($k in $c.e.Keys){ Set-Item "Env:\$k" $c.e[$k] }
  $dir = "D:\TaoVm\build\sc3_run"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue
  Remove-Item -Force "$dir.log","$dir.out","$dir.err" -EA SilentlyContinue
  $t0 = Get-Date
  $p = Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList @("build\probe_one","build\tok_v2.bbp",$dir,"$STEPS","16","16") -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $peak = 0
  while(-not $p.HasExited){
    $m = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if($m -gt $peak){ $peak = $m }
    Start-Sleep -Milliseconds 400
  }
  $dt = ((Get-Date) - $t0).TotalSeconds
  $upd = 0
  if(Test-Path "$dir.log"){ $upd = (Select-String -Path "$dir.log" -Pattern "^UPDATE " -EA SilentlyContinue | Measure-Object).Count }
  if($upd -gt 0){
    $line = ("{0,-30} 步={1,3} 用时={2,7:N1}s 每步={3,6:N3}s {4,8:N1} 位置/s  显存峰值={5,5} MiB" -f $c.n,$upd,$dt,($dt/$upd),(($upd*16*16)/$dt),($peak-$base))
  } else {
    $tail = ((Get-Content "$dir.out","$dir.err" -Tail 2 -EA SilentlyContinue) -join " | ")
    $line = ("{0,-30} 失败 用时={1:N1}s  {2}" -f $c.n,$dt,$tail)
  }
  Write-Output $line; Add-Content -Path $OUT -Value $line -Encoding utf8
}
Write-Output "GPT1_SCALE_DONE"; Add-Content -Path $OUT -Value "GPT1_SCALE_DONE" -Encoding utf8
