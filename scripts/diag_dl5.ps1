$ErrorActionPreference="Continue"
Write-Output "=== 端口 8889 / 7890 作为 HTTP 代理测试下载端点 ==="
$url="https://hf-mirror.com/datasets/codeparrot/codeparrot-valid/resolve/main/train.json.gz"
foreach($px in @("http://127.0.0.1:7890","http://127.0.0.1:8889")){
  $w=(& curl.exe -sSL --max-time 25 -x $px -r 0-65535 -o NUL -w "%{http_code} %{size_download} %{speed_download}" $url 2>&1|Out-String).Trim()
  Write-Output ("  HTTP-proxy {0,-24} {1}" -f $px,$w)
  $w2=(& curl.exe -sSL --max-time 25 --socks5-hostname $px -r 0-65535 -o NUL -w "%{http_code} %{size_download}" $url 2>&1|Out-String).Trim()
  Write-Output ("  SOCKS5     {0,-24} {1}" -f $px,$w2)
}
Write-Output "=== 直连官方 resolve ==="
$w3=(& curl.exe -sSL --max-time 25 -r 0-65535 -o NUL -w "%{http_code} %{size_download}" "https://huggingface.co/datasets/codeparrot/codeparrot-valid/resolve/main/train.json.gz" 2>&1|Out-String).Trim()
Write-Output ("  官方直连: " + $w3)
Write-Output "=== hf CLI 禁用 Xet 后下载小文件 ==="
$env:HF_HUB_DISABLE_XET="1"
$env:HF_ENDPOINT="https://hf-mirror.com"
Remove-Item -Recurse -Force E:\taovm-data\_t2 -EA SilentlyContinue
$r=(& hf download shibing624/alpaca-zh --repo-type dataset --local-dir E:\taovm-data\_t2 --include "README.md" 2>&1|Out-String)
Write-Output ("  " + $r.Substring(0,[Math]::Min(500,$r.Length)))
Get-ChildItem E:\taovm-data\_t2 -Recurse -File -EA SilentlyContinue | Select-Object Name,Length | Format-Table -AutoSize
Write-Output "=== 当前语料的原始下载记录 ==="
Get-Content D:\TaoVm\build\dl_code.log -EA SilentlyContinue | Select-Object -First 12
Get-Content E:\taovm-data\wmi_test.txt -EA SilentlyContinue | Select-Object -First 12
