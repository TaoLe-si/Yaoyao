$ProgressPreference='SilentlyContinue'
$repos = @(
  @{id="codeparrot/codeparrot-clean";                    t="英文代码 全量"},
  @{id="opencsg/chinese-fineweb-edu";                    t="中文通用"},
  @{id="pleisto/wikipedia-cn-20230720-filtered";         t="中文维基"},
  @{id="m-a-p/Matrix";                                    t="中英混合"},
  @{id="BAAI/CCI3-HQ";                                    t="中文高质量"},
  @{id="wikimedia/wikipedia";                             t="多语维基"},
  @{id="HuggingFaceTB/finemath";                          t="数学"},
  @{id="open-r1/OpenR1-Math-220k";                        t="数学推理"}
)
foreach($r in $repos){
  $u = "https://hf-mirror.com/api/datasets/" + $r.id + "?blobs=true"
  $raw = & curl.exe -sL --max-time 45 $u 2>$null | Out-String
  if($raw.Length -lt 50){ Write-Output ("{0,-46} 无响应" -f $r.id); continue }
  try{
    $j = $raw | ConvertFrom-Json
    $sib = @($j.siblings)
    $tot = 0; foreach($s in $sib){ if($s.size){ $tot += [long]$s.size } }
    $grp = $sib | ForEach-Object { [System.IO.Path]::GetExtension($_.rfilename) } | Where-Object {$_} | Group-Object | Sort-Object Count -Descending | Select-Object -First 3
    $ext = ($grp | ForEach-Object { $_.Name + "x" + $_.Count }) -join " "
    Write-Output ("{0,-46} {1,-12} 文件={2,5} 合计={3,8:N1} GB  格式: {4}" -f $r.id,$r.t,$sib.Count,($tot/1GB),$ext)
  }catch{ Write-Output ("{0,-46} 解析失败: {1}" -f $r.id,$_.Exception.Message) }
}