# 标度扫描：几何固定 sl=8 wd=8，d 递增，看 tok/s 的衰减指数
$ErrorActionPreference="Continue"; Set-Location D:\TaoVm
$env:TAO_ALLOW_TOKENIZER="1"; $env:TAO_CPU_THREADS="4"; $env:TAO_CFG_LAYERS="2"
$OUT="D:\TaoVm\build\scale.log"; Set-Content $OUT "scaling" -Encoding utf8
$i=0
function B($d){
  $script:i++
  $dk=[int]($d/8); $s=[int]($d/2)
  Set-Item Env:\TAO_CFG_D $d; Set-Item Env:\TAO_CFG_S $s; Set-Item Env:\TAO_CFG_M $d; Set-Item Env:\TAO_CFG_DK $dk
  $dir="D:\TaoVm\build\sc_$($script:i)"
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue; Remove-Item -Force "$dir.log" -EA SilentlyContinue
  $ar=@("build\probe_one","build\tok_v2.bbp",$dir,"4","8","8")
  $t0=Get-Date
  $p=Start-Process -FilePath ".\build\train_shards.exe" -ArgumentList $ar -PassThru -NoNewWindow -RedirectStandardOutput "$dir.out" -RedirectStandardError "$dir.err"
  $dl=(Get-Date).AddSeconds(600)
  while(-not $p.HasExited -and (Get-Date) -lt $dl){Start-Sleep -Milliseconds 300}
  if(-not $p.HasExited){$p.Kill();$p.WaitForExit()}
  $dt=((Get-Date)-$t0).TotalSeconds; $tot=0;$n=0
  if(Test-Path "$dir.log"){foreach($ln in (Get-Content "$dir.log" -EA SilentlyContinue)){if($ln -match "^UPDATE " -and $ln -match "positions=(\d+)"){$tot+=[long]$matches[1];$n++}}}
  $P=(16384*$d+16384+$d)+2*(2*($s*$d+$s*$s+$s)+($dk*$d*2+$d*$d+$d+1)+$d*$s+2*$d)
  if($n -ge 1){
    $tps=$tot/$dt; $line=("d={0,5} dk={1,4} s={2,5} 参数={3,6:N1}M 步数={4} t/步={5,5:N2}s tok/s={6,8:N0} 占比/参数={7,11:N4}" -f $d,$dk,$s,($P/1e6),$n,($dt/$n),$tps,($tps/($P/1e6)*1000))
  }else{$line=("d={0,5} FAIL {1:N1}s" -f $d,$dt)}
  Write-Output $line; Add-Content $OUT $line -Encoding utf8
  Remove-Item -Recurse -Force $dir -EA SilentlyContinue }
B 1024
B 1536
B 2048
B 2560
B 3200
Write-Output "SCALE_DONE"; Add-Content $OUT "SCALE_DONE" -Encoding utf8
