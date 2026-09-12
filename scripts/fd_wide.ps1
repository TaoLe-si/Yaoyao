# 在加宽配置下做同一套有限差分校验（delta 内核的并行度改动必须逐尺寸验证）
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$OUT="D:\TaoVm\build\fd_wide.log"; Set-Content $OUT "finite difference at wide configs" -Encoding utf8
$env:TAO_CFG_LAYERS="1"
foreach($cfg in @(@(512,256,512,64),@(1024,512,1024,128),@(2048,1024,2048,256))){
  $d=$cfg[0];$s=$cfg[1];$m=$cfg[2];$dk=$cfg[3]
  Set-Item Env:\TAO_CFG_D $d; Set-Item Env:\TAO_CFG_S $s; Set-Item Env:\TAO_CFG_M $m; Set-Item Env:\TAO_CFG_DK $dk
  Set-Item Env:\TAO_CFG_VOCAB "2048"
  $r = & .\build\test_delta_grad.exe 3 2>&1 | Out-String
  $verdict = if ($r -match "结论: 通过") { "PASS" } elseif ($r -match "结论: 失败") { "FAIL" } else { "??" }
  $worst = if ($r -match "最差超标倍数 = ([0-9.]+)") { $matches[1] } else { "?" }
  $nok = ([regex]::Matches($r,"OK")).Count
  $badline = ($r -split "`n" | Where-Object {$_ -match "结论"} | Select-Object -First 1)
  $l = ("d={0,5} s={1,5} m={2,5} dk={3,4}  OK数={4,3}  最差超标倍数={5}  {6}" -f $d,$s,$m,$dk,$nok,$worst,$verdict)
  Write-Output $l; Add-Content $OUT $l -Encoding utf8 }
Write-Output "FD_WIDE_DONE"; Add-Content $OUT "FD_WIDE_DONE" -Encoding utf8
