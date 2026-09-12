$ErrorActionPreference="Continue"
$UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
function T($tag,$url,$extra){
  $args=@("-sSL","--max-time","40","-o","NUL","-w","%{http_code} %{size_download} %{speed_download}")
  if($extra){$args+=$extra}
  $args+=$url
  $w=(& curl.exe @args 2>&1|Out-String).Trim()
  Write-Output ("{0,-46} {1}" -f $tag,$w)
}
Write-Output "=== A. 小文件，不带 range ==="
T "hf-mirror README" "https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/README.md" @()
T "hf-mirror alpaca README" "https://hf-mirror.com/datasets/shibing624/alpaca-zh/resolve/main/README.md" @()
T "hf-mirror codeparrot gitattr" "https://hf-mirror.com/datasets/codeparrot/codeparrot-valid/resolve/main/.gitattributes" @()
Write-Output "=== B. 带 ?download=true ==="
T "CCI3 ?download" "https://hf-mirror.com/datasets/BAAI/CCI3-HQ/resolve/main/data/part_000000.jsonl?download=true" @("-r","0-65535")
Write-Output "=== C. 代理端口探测 ==="
foreach($p in @(7890,7897,10809,1080,8889)){
  $ok=(Test-NetConnection -ComputerName 127.0.0.1 -Port $p -WarningAction SilentlyContinue).TcpTestSucceeded
  Write-Output ("  port " + $p + " = " + $ok)
}
Write-Output "=== D. 代理直接请求（看返回体）==="
$r=(& curl.exe -sS --max-time 25 -x "http://127.0.0.1:7890" "https://hf-mirror.com/api/datasets/codeparrot/codeparrot-clean?blobs=true" 2>&1|Out-String)
Write-Output ("  长度=" + $r.Length); Write-Output ("  头部=" + $r.Substring(0,[Math]::Min(200,$r.Length)))
Write-Output "=== E. 无代理 API（已知可用）==="
$r2=(& curl.exe -sS --max-time 25 "https://hf-mirror.com/api/datasets/codeparrot/codeparrot-clean?blobs=true" 2>&1|Out-String)
Write-Output ("  长度=" + $r2.Length)
Write-Output "=== F. hf CLI 端点识别 ==="
$env:HF_ENDPOINT="https://hf-mirror.com"
& hf env 2>&1 | Select-Object -First 20
