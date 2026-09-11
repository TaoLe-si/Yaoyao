// 记忆化判定：F1 + 8-gram 逐字覆盖 + 前缀匹配长度。
// 用法: node scripts/eval_memo.mjs <dsb> <jsonl> [n]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
const DSB=process.argv[2], HL=process.argv[3], N=Number(process.argv[4]||200);
const recs=fs.readFileSync(HL,'utf8').split('\n').filter(Boolean).slice(0,N).map(l=>JSON.parse(l));
const input=[]; for(const r of recs){input.push('/reset');input.push(r.user);} input.push('/quit');
const p=spawnSync('D:/TaoVm/build/h2r_cpu.exe',[DSB],{cwd:'D:/TaoVm',input:input.join('\n')+'\n',encoding:'utf8',
 timeout:1800000,maxBuffer:1<<27,env:{...process.env,TAO_TOKENIZER:'D:/TaoVm/build/tok_real_v1.bbp',TAO_CPU_THREADS:'8'}});
const outs=[...String(p.stdout||'').matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map(m=>m[1]);
const norm=s=>s.replace(/\s+/g,'');
const grams=(s,n)=>{const o=new Set();for(let i=0;i+n<=s.length;i++)o.add(s.slice(i,i+n));return o;};
let f1s=0, covs=0, prefs=0, n=0;
for(let i=0;i<Math.min(recs.length,outs.length);i++){
  const o=norm(outs[i]), r=norm(recs[i].ref); if(!r) continue; n++;
  const go=grams(o,2), gr=grams(r,2); let hit=0; for(const g of gr) if(go.has(g)) hit++;
  const prec=hit/ (go.size||1), rec=hit/(gr.size||1);
  f1s += (prec+rec)? 2*prec*rec/(prec+rec) : 0;
  const g8o=grams(o,8), g8r=grams(r,8); let h8=0; for(const g of g8r) if(g8o.has(g)) h8++;
  covs += g8r.size? h8/g8r.size : 0;
  let k=0; while(k<Math.min(o.length,r.length)&&o[k]===r[k]) k++;
  prefs += k;
}
console.log(DSB.split('/').slice(-2).join('/')+'\tn='+n+'\tF1='+(f1s/n).toFixed(4)+'\t8gramCoverage='+(covs/n).toFixed(4)+'\t前缀匹配='+(prefs/n).toFixed(1)+'字');
