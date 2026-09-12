$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_TOKENIZER="D:/TaoVm/build/tok_real_v1.bbp"
$env:TAO_CPU_THREADS="6"
$lines=@()
foreach($rp in @("1.15","1.30","1.50","1.80","2.20")){
  $out = & .\build\grpo_rollout.exe build/s2_night1/step_50659/final.dsb E:/taovm-data/real_reason/ape210k_test_probe.jsonl --n 30 --samples 8 --temp 1.0 --top-p 0.98 --max 256 --rep-pen $rp --show 0 2>&1
  $sel = $out | Select-String "平均生成长度|pass@|格式率|组内方差"
  $lines += "######## rep_pen=$rp"
  foreach($s in $sel){ $lines += ("  " + $s.ToString().Trim()) }
}
$lines | Out-File -Encoding utf8 build\rep_sweep.log
Write-Output "SWEEP_DONE"
