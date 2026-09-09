param([string]$Checkpoint = 'build/yaoyao_train_step_31.scp')
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot ..)
$lock = $null
try {
  $lock = [IO.File]::Open((Join-Path $PSScriptRoot 'build/training-runner.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
  if (Get-Process -Name train_yaoyao,train_yaoyao_continuous -ErrorAction SilentlyContinue) { throw 'Trainer already running' }
  if (!(Test-Path $Checkpoint)) { throw 'Checkpoint missing' }
  if (Test-Path 'build/STOP_TRAINING') { throw 'STOP_TRAINING exists; inspect before removing' }
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
  $log = 'build/continuous_' + $stamp + '.log'
  Write-Host "Training in this terminal; log: $log. Create build/STOP_TRAINING for safe boundary stop."
  & '.\build\train_yaoyao_continuous.exe' $Checkpoint *> $log
  if ($LASTEXITCODE -ne 0) { throw "Training failed; inspect $log; no automatic retry" }
  Write-Host "Trainer exited successfully; inspect $log for final saved step."
} finally {
  if ($lock) { $lock.Dispose() }
}
