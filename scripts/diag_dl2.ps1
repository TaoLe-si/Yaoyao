$ErrorActionPreference="Continue"
$UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
$u1="https://hf-mirror.com/BAAI/CCI3-HQ/resolve/main/data/part_000000.jsonl"
$u2="https://huggingface.co/datasets/BAAI/CCI3-HQ/resolve/main/data/part_000000.jsonl"
$u3="https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/data.json"
function T($tag,$url,$extra){
  $tmp="D:\TaoVm\build\dt.bin"
  Remove-Item -Force $tmp -EA SilentlyContinue
  $a=@("-sL","--max-time","25","-r","0-1048575","-o",$tmp,"-w","%{http_code} %{size_download} %{speed_download}")
  if($extra){ $a += $extra }
  $a += $url
  $w = & curl.exe @a 2>&1 | Out-String
  Write-Output ("{0,-38} {1}" -f $tag,($w.Trim()))
  Remove-Item -Force $tmp -EA SilentlyContinue
}
Write-Output "=== 无代理 ==="
T "mirror 无UA"      $u1 $null
T "mirror 带UA"      $u1 @("-A",$UA)
T "mirror 带UA+refer" $u1 @("-A",$UA,"-e","https://hf-mirror.com/")
T "官方 无UA"        $u2 $null
T "mirror 维基小文件" $u3 @("-A",$UA)
Write-Output "=== 带代理 7890 ==="
$env:HTTPS_PROXY="http://127.0.0.1:7890"; $env:HTTP_PROXY="http://127.0.0.1:7890"
T "官方 带代理"      $u2 @("-A",$UA)
T "mirror 带代理"    $u1 @("-A",$UA)
Remove-Item Env:\HTTPS_PROXY,Env:\HTTP_PROXY -EA SilentlyContinue
Write-Output "=== 403 响应体 ==="
& curl.exe -sL --max-time 20 -r 0-2000 $u1 2>&1 | Out-String | ForEach-Object { $_.Substring(0,[Math]::Min(400,$_.Length)) }
Write-Output "=== 现有语料来源线索 ==="
Get-Content D:\TaoVm\scripts\test_mirror.bat -EA SilentlyContinue
Get-ChildItem D:\TaoVm\build\dl_*.log,E:\taovm-data\*.log -EA SilentlyContinue | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
