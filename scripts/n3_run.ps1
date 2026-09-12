cd D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="8"; $env:TAO_SHUFFLE_SEED="20260922"
$env:TAO_LR="0.0002"; $env:TAO_LR_DECAY_START="59768"; $env:TAO_LR_DECAY_STEPS="29415"; $env:TAO_LR_MIN="0.00001"
if(Test-Path Env:\TAO_CONVERGE){ Remove-Item Env:\TAO_CONVERGE }
$plan=@( @("n3_reason",660,"build/s2_conv3","build/n3_a"), @("n3_alpaca",550,"build/n3_a","build/n3_b"), @("n3_wiki",1975,"build/n3_b","build/n3_c") )
foreach($s in $plan){
  $d=$s[0]; $st=$s[1]; $res=$s[2]; $out=$s[3]
  if(Test-Path $out){ Remove-Item -Recurse -Force $out }
  if(Test-Path "$out.log"){ Remove-Item -Force "$out.log" }
  Write-Output ("=== START " + $d + " steps=" + $st + " resume=" + $res + " out=" + $out + "  " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  & .\build\train_shards.exe "data/$d" build/tok_real_v1.bbp $out $st 32 32 $res 0 2>&1 | Out-Null
  Write-Output ("=== DONE " + $out + " exit=" + $LASTEXITCODE + "  " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
}
Write-Output "N3_ALL_DONE"