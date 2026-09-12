cd D:\TaoVm
# 等长训结束（连续 3 次探测无 train_shards 进程即认为结束；上限 16 小时）
$deadline=(Get-Date).AddHours(16); $idle=0
while((Get-Date) -lt $deadline){
  $p=Get-Process train_shards -ErrorAction SilentlyContinue
  if($p){ $idle=0 } else { $idle++; if($idle -ge 3){ break } }
  Start-Sleep -Seconds 60
}
Write-Output ("训练结束于 " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " idle=$idle")
Get-Content build\s2_conv2.log | Select-String "SHARD_CONVERGED|SHARD_CAPPED|SHARD_OVERFIT|PAUSED|EXIT" | Select-Object -Last 30
Write-Output ""; Write-Output "=== 开始质量验证 ==="
& node scripts/morning_test.mjs 400 2>&1 | Out-File -Encoding utf8 build\conv_quality_run.log
Write-Output ("morning_test exit=" + $LASTEXITCODE)
Write-Output "REPORT -> build/MORNING_REPORT.md"
Write-Output "CONV_FINALIZE_DONE"