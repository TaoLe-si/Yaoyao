$ErrorActionPreference="Continue"
$UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
function T($tag,$url,$extra){
  $tmp="D:\TaoVm\build\dt2.bin"
  Remove-Item -Force $tmp -EA SilentlyContinue
  $a=@("-sL","--max-time","30","-r","0-1048575","-o",$tmp,"-w","%{http_code} %{size_download} %{speed_download}")
  if($extra){$a+=$extra}
  $a+=$url
  $w=(& curl.exe @a 2>&1|Out-String).Trim()
  Write-Output ("{0,-44} {1}" -f $tag,$w)
  Remove-Item -Force $tmp -EA SilentlyContinue
}
Write-Output "=== 正确 datasets 路径 ==="
T "CCI3-HQ (需授权?)" "https://hf-mirror.com/datasets/BAAI/CCI3-HQ/resolve/main/data/part_000000.jsonl" @("-A",$UA)
T "codeparrot-clean train" "https://hf-mirror.com/datasets/codeparrot/codeparrot-clean/resolve/main/train/0000.parquet" @("-A",$UA)
T "codeparrot-clean valid" "https://hf-mirror.com/datasets/codeparrot/codeparrot-clean-valid/resolve/main/train.json.gz" @("-A",$UA)
T "wikipedia-cn (pleisto)" "https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/data.json" @("-A",$UA)
T "chinese-fineweb-edu" "https://hf-mirror.com/datasets/opencsg/chinese-fineweb-edu/resolve/main/data/part-00000.parquet" @("-A",$UA)
Write-Output "=== 已有语料的上游线索 ==="
Get-ChildItem E:\taovm-data\*.txt,E:\taovm-data\*.log -EA SilentlyContinue | Select-Object Name,Length | Format-Table -AutoSize
Write-Output "--- mirror_test.txt ---"
Get-Content E:\taovm-data\mirror_test.txt -EA SilentlyContinue | Select-Object -First 8
Write-Output "--- wiki_cn.json 头部（判断格式与来源）---"
$fs=[System.IO.File]::OpenRead("E:\taovm-data\wiki_cn.json")
$b=New-Object byte[] 600; $n=$fs.Read($b,0,600); $fs.Close()
Write-Output ([System.Text.Encoding]::UTF8.GetString($b,0,$n))
Write-Output "--- code_docs.txt 头部 ---"
$fs2=[System.IO.File]::OpenRead("E:\taovm-data\code_docs.txt")
$b2=New-Object byte[] 400; $n2=$fs2.Read($b2,0,400); $fs2.Close()
Write-Output ([System.Text.Encoding]::UTF8.GetString($b2,0,$n2))
