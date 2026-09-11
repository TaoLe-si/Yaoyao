// 长度无关的重复度与相关性：判定"重复率下降"是否为长度缩短的机械后果。
// 用法: node scripts/eval_repeat.mjs <dsb> [n]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
const DSB=process.argv[2], N=Number(process.argv[3]||400);
const recs=fs.readFileSync('D:/TaoVm/data/alpaca_heldout.jsonl','utf8').split('\n').filter(Boolean).slice(0,N).map(l=>JSON.parse(l));
const input=[]; for(const r of recs){input.push('/reset');input.push(r.user);} input.push('/quit');
const p=spawnSync('D:/TaoVm/build/h2r_cpu.exe',[DSB],{cwd:'D:/TaoVm',input:input.join('\n')+'\n',encoding:'utf8',
 timeout:1800000,maxBuffer:1<<27,env:{...process.env,TAO_TOKENIZER:'D:/TaoVm/build/tok_real_v1.bbp',TAO_CPU_THREADS:'8'}});
const bodies=[...String(p.stdout||'').matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map(m=>m[1]);
const norm=s=>s.replace(/\s+/g,'');
const ng=(s,n)=>{const o=[];for(let i=0;i+n<=s.length;i++)o.push(s.slice(i,i+n));return o;};
const isLoop=s=>{if(s.length<24)return false;for(let L=8;L<=16;L++)for(let i=0;i+L<=s.length;i++){const seg=s.slice(i,i+L);let c=0,j=i;while((j=s.indexOf(seg,j))!==-1){c++;j+=L;if(c>=3)return true;}}return false;};
function selfRep(s,n){const g=ng(s,n);if(!g.length)return 0;const d=new Set(g).size;return 1-d/g.length;}
function maxDup(s,n){const g=ng(s,n);if(!g.length)return 0;const m=new Map();let best=0;for(const x of g){const c=(m.get(x)||0)+1;m.set(x,c);if(c>best)best=c;}return best*n/s.length;}
let sumLen=0,sumR4=0,sumR8=0,sumDup=0,sumF1=0,sumPrec=0,sumRec=0,loopAll=0,loopElig=0,elig=0,n=0;
const rows=[];
for(let i=0;i<recs.length;i++){
  const got=norm(bodies[i]||''), ref=norm(recs[i].ref||''); if(!got||!ref)continue; n++;
  const r4=selfRep(got,4), r8=selfRep(got,8), dup=maxDup(got,8);
  const G=new Set(ng(got,2)), F=new Set(ng(ref,2));
  let inter=0; for(const x of G) if(F.has(x)) inter++;
  const f1=(G.size&&F.size)?2*inter/(G.size+F.size):0;
  const prec=G.size?inter/G.size:0, rec=F.size?inter/F.size:0;
  const lp=isLoop(got); if(lp)loopAll++; if(got.length>=24){elig++; if(lp)loopElig++;}
  sumLen+=got.length; sumR4+=r4; sumR8+=r8; sumDup+=dup; sumF1+=f1; sumPrec+=prec; sumRec+=rec;
  rows.push([got.length,r4]);
}
// len 与 selfRep4 的相关系数（看重复度是否只是长度的函数）
const mL=sumLen/n, mR=sumR4/n;
let cov=0,vL=0,vR=0; for(const [l,r] of rows){cov+=(l-mL)*(r-mR);vL+=(l-mL)**2;vR+=(r-mR)**2;}
const corr=(vL&&vR)?cov/Math.sqrt(vL*vR):0;
console.log([DSB.split('/').slice(-2)[0],'n='+n,'均长='+mL.toFixed(0),'自重复4='+mR.toFixed(4),'自重复8='+(sumR8/n).toFixed(4),'最重复8占比='+(sumDup/n).toFixed(3),
 'F1='+(sumF1/n).toFixed(4),'精确率='+(sumPrec/n).toFixed(4),'召回='+(sumRec/n).toFixed(4),'loop总='+loopAll,'loop(>=24字)='+loopElig+'/'+elig,'corr(len,rep)='+corr.toFixed(2)].join('\t'));
