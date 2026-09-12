$ErrorActionPreference="Continue"
Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"
Remove-Item Env:\TAO_OPT_OFFLOAD -EA SilentlyContinue
$env:TAO_CFG_LAYERS="2"; $env:TAO_CFG_D="3072"; $env:TAO_CFG_S="1536"; $env:TAO_CFG_M="3072"; $env:TAO_CFG_DK="384"
Remove-Item -Recurse -Force build\san2 -EA SilentlyContinue
& compute-sanitizer --tool memcheck --launch-timeout 240 --print-limit 30 .\build\train_shards.exe build\probe_one build\tok_v2.bbp build\san2 1 8 8 *>&1 | Out-File -Encoding utf8 build\san2.txt
Write-Output ("SAN2_EXIT=" + $LASTEXITCODE)
Get-Content build\san2.log -EA SilentlyContinue
