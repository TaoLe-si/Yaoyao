$ErrorActionPreference="Continue"
Write-Output "=== 1) cci3hq 目录 ==="
Get-ChildItem E:\taovm-data\cci3hq -Recurse -File -EA SilentlyContinue | Select-Object Name,Length | Format-Table -AutoSize
Write-Output "=== 2) 无代理直连测速（200MB range, 上限30s）==="
$url="https://hf-mirror.com/datasets/BAAI/CCI3-HQ/resolve/main/data/part_000000.jsonl"
$tmp="D:\TaoVm\build\cci_test.bin"
Remove-Item -Force $tmp -EA SilentlyContinue
$w = & curl.exe -sL --max-time 30 -r 0-209715199 -o $tmp -w "%{speed_download} %{size_download} %{http_code}" $url 2>&1
Write-Output ("直连: " + $w)
Remove-Item -Force $tmp -EA SilentlyContinue
Write-Output "=== 3) 带代理测速 ==="
$env:HTTPS_PROXY="http://127.0.0.1:7890"; $env:HTTP_PROXY="http://127.0.0.1:7890"
$w2 = & curl.exe -sL --max-time 30 -r 0-209715199 -o $tmp -w "%{speed_download} %{size_download} %{http_code}" $url 2>&1
Write-Output ("代理: " + $w2)
Remove-Item -Force $tmp -EA SilentlyContinue
Remove-Item Env:\HTTPS_PROXY -EA SilentlyContinue; Remove-Item Env:\HTTP_PROXY -EA SilentlyContinue
Write-Output "=== 4) NotReady 来源 ==="
Select-String -Path D:\TaoVm\src\*.cuh,D:\TaoVm\src\*.hpp,D:\TaoVm\src\*.cu -Pattern "cudaEventQuery|cudaStreamQuery|cudaErrorNotReady" -EA SilentlyContinue | ForEach-Object { $_.Filename + ":" + $_.LineNumber + "  " + $_.Line.Trim() }
Write-Output "=== 5) 选型探针 ==="
Get-Content D:\TaoVm\build\gpt1_scale.log -EA SilentlyContinue