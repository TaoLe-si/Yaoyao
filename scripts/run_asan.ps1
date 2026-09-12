$dll = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\18" -Recurse -Filter clang_rt.asan_dynamic-x86_64.dll -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$dir = Split-Path $dll
Copy-Item $dll -Destination D:\TaoVm\build\ -Force
$env:ASAN_OPTIONS = "detect_leaks=0"
$env:PATH = $dir + ";" + $env:PATH
Set-Location D:\TaoVm
& .\build\eval_nll_asan.exe --model build\L1_pretrain\step_900\final.dsb --text data\inmap_wiki.txt --tokenizer build\tok_v2.bbp --threads 1 --max-tokens 50 > build\asan_out.txt 2> build\asan_err.txt
Write-Output ("exit=" + $LASTEXITCODE)
Write-Output "--- stdout ---"; Get-Content build\asan_out.txt -EA SilentlyContinue | Select-Object -First 10
Write-Output "--- stderr ---"; Get-Content build\asan_err.txt -EA SilentlyContinue | Select-Object -First 45
