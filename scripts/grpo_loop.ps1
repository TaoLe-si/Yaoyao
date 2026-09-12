# GRPO 迭代循环：采样 -> 规则奖励 -> 组内优势 -> 加权策略梯度更新。
# 每轮从"当前策略"重新采样（on-policy），这是 GRPO 与一次性 SFT 的本质区别。
param(
  [string]$StartModel = "D:\\TaoVm\\build\\n3_c",
  [int]$Rounds = 6,
  [int]$Probs = 60,
  [int]$Samples = 16,
  [int]$Updates = 20,
  [double]$Lr = 2e-5,
  [string]$Tag = "g"
)
$ErrorActionPreference = "Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER = "1"
$env:TAO_TOKENIZER = "D:/TaoVm/build/tok_real_v1.bbp"
$env:TAO_CPU_THREADS = "6"
$PROBE = "D:/TaoVm/data/grpo_train.jsonl"   # 训练提示（202,652 题），与测试集零重叠
$EVAL  = "D:/TaoVm/data/grpo_test.jsonl"    # 仅用于评测（3,104 题），绝不参与训练
$LOG = "D:\TaoVm\build\grpo_loop.log"
function Say($m){ $s = (Get-Date -Format "HH:mm:ss") + "  " + $m; Write-Output $s; Add-Content -Path $LOG -Value $s -Encoding utf8 }
Say "==== GRPO 循环开始 start=$StartModel rounds=$Rounds ===="
$cur = $StartModel
for($r = 1; $r -le $Rounds; $r++){
  $base = "D:\TaoVm\build\${Tag}_r$r"
  $out  = "${base}_model"
  $traj = "${base}_traj.tsv"
  if(Test-Path $out){ Say "轮 $r 输出已存在，跳过"; $cur = $out; continue }
  $dsb = Join-Path $cur "final.dsb"
  if(-not (Test-Path $dsb)){ $d = Get-ChildItem (Join-Path $cur "step_*") -Directory | Sort-Object Name | Select-Object -Last 1; $dsb = Join-Path $d.FullName "final.dsb" }
  Say "轮 $r 采样源自 $dsb"
  & .\build\grpo_rollout.exe $dsb $PROBE --n $Probs --samples $Samples --temp 1.1 --top-p 0.98 --max 256 --show 0 --dump $traj 2>&1 | Out-File -Encoding utf8 "${base}_roll.log"
  $sig = Select-String -Path "${base}_roll.log" -Pattern "pass@|组内方差" | ForEach-Object { $_.Line.Trim() }
  foreach($s in $sig){ Say "   $s" }
  $ndocs = (Get-Content $traj | Measure-Object -Line).Lines
  if($ndocs -lt 2){ Say "轮 $r 无可用轨迹，跳过更新"; continue }
  & .\build\train_grpo.exe $traj build/tok_real_v1.bbp $out $Updates $cur 16 256 $Lr 2>&1 | Out-Null
  if(Test-Path (Join-Path $out "final.dsb")){ Say "轮 $r 完成 -> $out"; $cur = $out } else { Say "轮 $r 训练失败，保持 $cur" }
}
Say "==== GRPO 循环结束 final=$cur ===="
"GRPO_DONE model=$cur" | Add-Content -Path $LOG -Encoding utf8
