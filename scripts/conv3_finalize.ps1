cd D:\TaoVm
$deadline=(Get-Date).AddHours(16); $idle=0
while((Get-Date) -lt $deadline){
  $p=Get-Process train_shards -ErrorAction SilentlyContinue
  if($p){ $idle=0 } else { $idle++; if($idle -ge 3){ break } }
  Start-Sleep -Seconds 60
}
Write-Output ("训练结束 " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Get-Content build\s2_conv3.log | Select-String "SHARD_CONVERGED|SHARD_CAPPED|SHARD_OVERFIT|PAUSED" | Select-Object -Last 30
& node scripts/morning_test.mjs 400 2>&1 | Out-File -Encoding utf8 build\conv3_quality_run.log
Write-Output ("morning_test exit=" + $LASTEXITCODE)
Write-Output "CONV3_FINALIZE_DONE"