import fs from 'node:fs';
const recs=fs.readFileSync('D:/TaoVm/data/alpaca_heldout.jsonl','utf8').split('\n').filter(Boolean).map(l=>JSON.parse(l));
const idx=[0,2,4,6,8,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78,82,86,90,94,98,102,106,110,114,118,122,126,130,134,138,142,146];
const pick=idx.map(i=>recs[i%recs.length]);
const MODEL=process.env.TAO_MODEL, PEN=process.env.TAO_PEN||'1.0';
let input=[];
for(const r of pick){input.push('/reset');input.push(r.user);}
input.push('/quit');
const {spawnSync}=await import('node:child_process');
const p=spawnSync('D:/TaoVm/build/h2r_cpu.exe',[MODEL],{
 cwd:'D:/TaoVm',input:input.join('\n')+'\n',encoding:'utf8',timeout:1200000,maxBuffer:1<<27,
 env:{...process.env,TAO_TOKENIZER:'D:/TaoVm/build/tok_real_v1.bbp',TAO_CPU_THREADS:'8',TAO_REP_PEN:PEN}});
const bodies=[...String(p.stdout||'').matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map(m=>m[1]);
console.log('==== '+MODEL.split('/').slice(-2).join('/')+'  pen='+PEN+'  n='+pick.length+' ====');
pick.forEach((r,i)=>{
 console.log('\n【Q'+(i+1)+'】'+r.user);
 console.log('【答】'+(bodies[i]||'(空)').slice(0,130));
});
