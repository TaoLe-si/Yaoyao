$ErrorActionPreference="Continue"
cd D:\TaoVm
$env:HTTPS_PROXY="http://127.0.0.1:7890"; $env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HF_ENDPOINT="https://hf-mirror.com"
Remove-Item -Recurse -Force E:\taovm-data\cci3hq -EA SilentlyContinue
New-Item -ItemType Directory -Force -Path E:\taovm-data\cci3hq | Out-Null
$t0=Get-Date
# 取 CCI3-HQ 前 16 个分片（每个约 1GB），中文高质量语料
$inc = @()
for($i=0;$i -lt 16;$i++){ $inc += ("data/part_" + $i.ToString("000000") + ".jsonl") }
$a = @("download","BAAI/CCI3-HQ","--repo-type","dataset","--local-dir","E:\taovm-data\cci3hq") + ($inc | ForEach-Object { "--include"; $_ })
& hf @a 2>&1 | Out-File -Encoding utf8 D:\TaoVm\build\dl_cci3.log
$dt=((Get-Date)-$t0).TotalSeconds
$sz=(Get-ChildItem E:\taovm-data\cci3hq -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
Write-Output ("CCI3_DONE seconds=" + [math]::Round($dt,0) + " bytes=" + $sz + " MB/s=" + [math]::Round($sz/1MB/$dt,1))
