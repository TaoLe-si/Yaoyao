$ErrorActionPreference="SilentlyContinue"
Write-Output "=== A. 代理端口重新确认 ==="
foreach($p in @(7890,8889,7897,10809,1080,8118,10808)){
  $ok=(Test-NetConnection -ComputerName 127.0.0.1 -Port $p -WarningAction SilentlyContinue).TcpTestSucceeded
  Write-Output ("  port " + $p + " = " + $ok)
}
Write-Output "=== B. 两个代理上测下载端点 ==="
foreach($px in @("http://127.0.0.1:7890","http://127.0.0.1:8889","socks5://127.0.0.1:7890")){
  $w=(& curl.exe -sSL --max-time 20 -x $px -r 0-65535 -o NUL -w "%{http_code} %{size_download}" "https://hf-mirror.com/datasets/codeparrot/codeparrot-valid/resolve/main/train.json.gz" 2>&1|Out-String).Trim()
  Write-Output ("  {0,-26} {1}" -f $px,$w)
}
Write-Output "=== C. 本机已有大规模语料候选（>50MB）==="
$roots=@("E:\taovm-data","E:\models","E:\FreeToken","E:\T","E:\taovm-tmp","D:\TaoVm\data")
$found=@()
foreach($r in $roots){
  if(-not (Test-Path $r)){continue}
  Get-ChildItem -Path $r -Recurse -File -Include *.jsonl,*.parquet,*.json.gz,*.gz,*.json,*.txt,*.bin -EA SilentlyContinue |
    Where-Object { $_.Length -gt 50MB } |
    ForEach-Object { $found += [PSCustomObject]@{MB=[math]::Round($_.Length/1MB,0); Path=$_.FullName} }
}
$found | Sort-Object MB -Descending | Select-Object -First 25 | Format-Table -AutoSize
Write-Output ("合计候选 = " + $found.Count)
Write-Output "=== D. E: 顶层非空目录大小前 15 ==="
Get-ChildItem E:\ -Directory -EA SilentlyContinue | ForEach-Object {
  $s=(Get-ChildItem $_.FullName -Recurse -File -EA SilentlyContinue | Measure-Object -Property Length -Sum).Sum
  [PSCustomObject]@{GB=[math]::Round($s/1GB,2); Name=$_.Name}
} | Sort-Object GB -Descending | Select-Object -First 15 | Format-Table -AutoSize
