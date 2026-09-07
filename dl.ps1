$ProgressPreference = 'SilentlyContinue'
# Use system proxy
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$proxy = [System.Net.WebRequest]::GetSystemWebProxy()
$proxy.Credentials = [System.Net.CredentialCache]::DefaultCredentials
[System.Net.WebRequest]::DefaultWebProxy = $proxy

$urls = @(
  @{ url = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet'; out = 'D:\TaoVm\tinystories_train_0.parquet' },
  @{ url = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet'; out = 'D:\TaoVm\tinystories_val_0.parquet' }
)
foreach ($item in $urls) {
  $url = $item.url; $out = $item.out
  if (Test-Path $out) { $size = (Get-Item $out).Length; Write-Host "$out exists: $size"; continue }
  for ($try=1; $try -le 3; $try++) {
    Write-Host "attempt $try for $out"
    try {
      $wc = New-Object System.Net.WebClient
      $wc.Proxy = $proxy
      $wc.DownloadFile($url, $out)
      $wc.Dispose()
      $size = (Get-Item $out).Length
      Write-Host "downloaded: $size bytes"
      break
    } catch {
      Write-Host "err: $($_.Exception.Message)"
      Start-Sleep 3
    }
  }
}
Write-Host 'final:'
Get-ChildItem 'D:TaoVm*.parquet' | Format-Table Name, Length -AutoSize
