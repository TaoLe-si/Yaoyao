param([Parameter(Mandatory=$true)][string]$Checkpoint)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot ..)
$lock = $null
try {
  $lock = [IO.File]::Open((Join-Path $PSScriptRoot 'build/training-runner.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
  if (Get-Process -Name train_yaoyao,train_yaoyao_continuous,train_yaoyao_parallel -ErrorAction SilentlyContinue) { throw 'Existing trainer detected; no duplicate launch' }
  if (!(Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw 'Checkpoint missing' }
  if ([IO.Path]::GetExtension($Checkpoint) -ne '.scp') { throw 'Training requires SCP master checkpoint, not inference bundle' }
  if (Test-Path 'build/STOP_TRAINING') { throw 'STOP_TRAINING exists; inspect before deliberate restart' }
  $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($PSScriptRoot))
  if ($drive.AvailableFreeSpace -lt 10GB) { throw 'Require 10GiB free for 200-update checkpoint run' }
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
  $log = 'build/parallel_' + $stamp + '.log'
  Write-Host "200 additional optimizer updates from $Checkpoint; log $log"
  Write-Host 'Create build/STOP_TRAINING to save and stop at a complete update boundary.'
  & '.\build\train_yaoyao_parallel.exe' $Checkpoint *> $log
  $code = $LASTEXITCODE
  if ($code -ne 0) { throw "Trainer exit $code; inspect $log; no automatic retry" }
  Get-Content -LiteralPath $log -Tail 4
} finally {
  if ($lock) { $lock.Dispose() }
}
