cd D:\TaoVm
$env:TAO_TOKENIZER="D:/TaoVm/build/tok_real_v1.bbp"; $env:TAO_CPU_THREADS="8"; $env:TAO_REP_WIN="0"
Write-Output "########## 对照: 上一交付 s2_night1/step_50659 ##########"
$env:TAO_REP_PEN="0"
& node scripts/eval_real.mjs build/s2_night1/step_50659/final.dsb data/alpaca_heldout.jsonl 400 2>&1
$env:TAO_REP_PEN="1.3"
& node scripts/eval_real.mjs build/s2_night1/step_50659/final.dsb data/alpaca_heldout.jsonl 400 2>&1
$env:TAO_REP_PEN="1.0"
& node scripts/eval_math.mjs build/s2_night1/step_50659/final.dsb 100 2>&1
Write-Output "BASELINE_EVAL_DONE"