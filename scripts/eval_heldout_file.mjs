
// Runs the FIXED held-out verbatim set (data/heldout_verbatim.txt) against a model.
// The set was verified novel against BOTH data/rebalanced_v2 and data/noffn_l6,
// so old-corpus and new-corpus models are compared on identical, unseen inputs.
// Usage: node scripts/eval_heldout_file.mjs <dsb> [exe]
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
const DSB=process.argv[2];
const EXE=process.argv[3]||process.env.TAO_CPU_EXE||'D:/TaoVm/build/h2r_cpu.exe';
if(!DSB||!fs.existsSync(DSB)){console.log('usage: node scripts/eval_heldout_file.mjs <dsb> [exe]');process.exit(1);}
const cand=fs.readFileSync('D:/TaoVm/data/heldout_verbatim.txt','utf8').split(/\r?\n/).filter(x=>x.trim());
const inp=[];for(const s of cand)inp.push('/reset','请重复这句话：'+s);inp.push('/quit');
const r=spawnSync(EXE,[DSB],{cwd:'D:/TaoVm',input:inp.join('\n')+'\n',encoding:'utf8',timeout:1800000,maxBuffer:1<<26});
const reps=[];let c=null;
for(const ln of String(r.stdout||'').split(/\r?\n/)){
  if(ln.startsWith('BEGIN_REPLY ')){c=[];continue;}
  if(ln.startsWith('END_REPLY ')){if(c)reps.push(c.join(''));c=null;continue;}
  if(c&&ln!=='RESET')c.push(ln);
}
const rows=cand.map((s,i)=>{const got=(reps[i]||'').trim();return {exp:s,got,exact:got===s,first:!!got&&got[0]===s[0],mangled:got.includes('\uFFFD')};});
const exact=rows.filter(x=>x.exact).length;
const first=rows.filter(x=>x.first).length;
const mangled=rows.filter(x=>x.mangled).length;
console.log('  model: '+DSB);
console.log('  语料外原样复制 完全正确 = '+exact+'/'+rows.length+'   首字正确 = '+first+'/'+rows.length+'   含乱码 = '+mangled);
if(process.env.TAO_SHOW==='1'){
  for(const x of rows){
    const mark=x.exact?'✓':'✗';
    console.log('   '+mark+' 期望='+JSON.stringify(x.exp)+' 模型='+JSON.stringify(x.got).slice(0,70));
  }
}
