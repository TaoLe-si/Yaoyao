param(
  [int]$FromStep = 31,
  [int]$UntilStep = 40
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root
if ($FromStep -lt 1 -or $UntilStep -le $FromStep -or $UntilStep -gt 100) { throw 'Invalid step range' }
$lockPath = Join-Path $root 'build/training-runner.lock'
$lock = $null
$process = $null
try {
  $lock = [System.IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None')
  if (Get-Process -Name train_yaoyao -ErrorAction SilentlyContinue) { throw 'Existing trainer detected; no duplicate launch' }
  $checkpoint = Join-Path $root ('build/yaoyao_train_step_{0}.scp' -f $FromStep)
  if (!(Test-Path $checkpoint)) { throw 'Resume checkpoint missing' }
  for ($step = $FromStep + 1; $step -le $UntilStep; $step++) {
    if (Test-Path (Join-Path $root 'build/STOP_TRAINING')) { Write-Host 'Stopped at optimizer boundary'; break }
    $next = Join-Path $root ('build/yaoyao_train_step_{0}.scp' -f $step)
    $model = Join-Path $root ('build/yaoyao_train_step_{0}.dsb' -f $step)
    foreach ($path in @($next, $model, ($next + '.tmp'), ($model + '.tmp'))) {
      if (Test-Path $path) { throw ('Refusing to overwrite existing artifact: ' + $path) }
    }
    $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($root))
    if ($drive.AvailableFreeSpace -lt 2GB) { throw 'Insufficient disk space for next checkpoint' }
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $stdout = Join-Path $root ('build/managed_step{0}_{1}.log' -f $step, $stamp)
    $stderr = Join-Path $root ('build/managed_step{0}_{1}.err' -f $step, $stamp)
    $process = Start-Process -FilePath (Join-Path $root 'build/train_yaoyao.exe') -ArgumentList ('"' + $checkpoint + '"') -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -NoNewWindow
    Write-Host ('Running step {0}, PID {1}; log {2}' -f $step, $process.Id, $stdout)
    $process.WaitForExit()
    $process.Refresh()
    if ($process.ExitCode -ne 0) { throw ('Trainer failed. Inspect ' + $stderr + ' and ' + $stdout) }
    $last = Get-Content -LiteralPath $stdout -Tail 1
    if ($last -notmatch ('^SAVED step=' + $step + ' ') -or !(Test-Path $next) -or !(Test-Path $model)) { throw 'Missing saved-step evidence; do not retry automatically' }
    Write-Host $last
    $checkpoint = $next
    $process.Dispose()
    $process = $null
  }
} finally {
  if ($process) {
    if (!$process.HasExited) { $process.Kill(); $process.WaitForExit(); Write-Warning 'Interrupted current step; previous complete checkpoint preserved' }
    $process.Dispose()
  }
  if ($lock) { $lock.Dispose() }
}
