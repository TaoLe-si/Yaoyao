$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
$env:TAO_CFG_LAYERS="12"; $env:TAO_CFG_D="1536"; $env:TAO_CFG_S="768"; $env:TAO_CFG_M="1536"; $env:TAO_CFG_DK="192"
Remove-Item -Recurse -Force build\san1 -EA SilentlyContinue
& compute-sanitizer --tool memcheck --launch-timeout 180 --print-limit 20 .\build\train_shards_blas.exe build\probe_one build\tok_v2.bbp build\san1 1 8 8 *>&1 | Out-File -Encoding utf8 build\san1.txt
Write-Output ("SAN_EXIT=" + $LASTEXITCODE)
Get-Content build\san1.log -EA SilentlyContinue
