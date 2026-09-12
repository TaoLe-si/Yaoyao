$ProgressPreference="SilentlyContinue"
$raw = & curl.exe -sL --max-time 60 "https://hf-mirror.com/api/datasets/BAAI/CCI3-HQ?blobs=true" 2>$null | Out-String
$j = $raw | ConvertFrom-Json
$sib = @($j.siblings) | Where-Object { $_.rfilename -like "*.jsonl" }
Write-Output ("jsonl 文件数 = " + $sib.Count)
$sib | Select-Object -First 8 | ForEach-Object { Write-Output ("  " + $_.rfilename + "  " + [math]::Round([long]$_.size/1MB,0) + " MB") }
$tot=0; foreach($s in $sib){ $tot += [long]$s.size }
Write-Output ("合计 = " + [math]::Round($tot/1GB,1) + " GB")
Write-Output "=== 总表: 按大小分布 ==="
$g = $sib | ForEach-Object { [math]::Round([long]$_.size/1MB/100)*100 } | Group-Object | Sort-Object Name
$g | ForEach-Object { Write-Output ("  " + $_.Name + " MB  x" + $_.Count) }