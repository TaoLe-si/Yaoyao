cd D:\TaoVm
$env:TAO_TOKENIZER="D:/TaoVm/build/tok_real_v1.bbp"; $env:TAO_CPU_THREADS="8"; $env:TAO_REP_WIN="0"
Write-Output "########## 最新检查点 step_58950 ##########"
$env:TAO_REP_PEN="0"
Write-Output "=== n=400  rep_pen=0 ==="
& node scripts/eval_real.mjs build/s2_conv2/step_58950/final.dsb data/alpaca_heldout.jsonl 400 2>&1
$env:TAO_REP_PEN="1.3"
Write-Output "=== n=400  rep_pen=1.3 ==="
& node scripts/eval_real.mjs build/s2_conv2/step_58950/final.dsb data/alpaca_heldout.jsonl 400 2>&1
$env:TAO_REP_PEN="1.0"
Write-Output "=== 真实数学推理 Ape210K test (n=100) ==="
& node scripts/eval_math.mjs build/s2_conv2/step_58950/final.dsb 100 2>&1
$env:TAO_MODEL="build/s2_conv2/step_58950/final.dsb"; $env:TAO_PEN="1.3"; $env:TAO_REP_PEN="1.3"
Write-Output "=== 40 条完整问答 (pi=1.3) ==="
& node scripts/show_full.mjs 2>&1
Write-Output "LATEST_EVAL_DONE"