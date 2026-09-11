
// Full post-training evaluation for rebalanced_v2 (current architecture).
// Usage: node scripts/eval_v2_all.mjs <dsb>
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
const DSB = process.argv[2] || 'D:/TaoVm/build/rebal_v2/final.dsb';
const EXE = 'D:/TaoVm/build/h2r_cpu.exe';
const CORPUS = 'D:/TaoVm/data/rebalanced_v2/conversations.txt';

function replies(input){
  const r=spawnSync(EXE,[DSB],{cwd:'D:/TaoVm',input,encoding:'utf8',timeout:1800000,maxBuffer:1<<27});
  const out=[];let c=null;
  for(const ln of String(r.stdout||'').split(/\r?\n/)){
    if(ln.startsWith('BEGIN_REPLY ')){c=[];continue;}
    if(ln.startsWith('END_REPLY ')){if(c)out.push(c.join(''));c=null;continue;}
    if(c&&ln!=='RESET')c.push(ln);
  }
  return out;
}

const lines=fs.readFileSync(CORPUS,'utf8').split(/\r?\n/);
const docs=[];let cur=[];
for(const l of lines){ if(l==='DOC'){if(cur.length)docs.push(cur);cur=[];continue;} if(l.trim()==='')continue; cur.push(l); }
if(cur.length)docs.push(cur);

// 1) in-corpus verbatim
const verb=docs.filter(d=>d.length===2&&/^U 请重复这句话/.test(d[0]));
// 2) in-corpus ordered copy
const ord=docs.filter(d=>d.length===2&&/^U (请按顺序输出|按顺序复述|请依次说出)/.test(d[0]));
// 3) arithmetic
const ar=docs.filter(d=>d.length===2&&/^U .*等于多少/.test(d[0]));
// 4) binding (2-turn memory)
const bind=docs.filter(d=>d.length===4&&/^U 请记住/.test(d[0]));
const pick=(a,n)=>{const o=[];for(let i=0;i<n&&i<a.length;i++)o.push(a[Math.floor(i*a.length/n)]);return o;};

const runSet=(set,label,promptOf,expectOf)=>{
  if(!set.length){console.log('  '+label+': (no samples)');return;}
  const inp=[];for(const d of set)inp.push('/reset',promptOf(d));inp.push('/quit');
  const rep=replies(inp.join('\n')+'\n');
  let ok=0;
  const detail=[];
  set.forEach((d,i)=>{const got=(rep[i]||'').trim();const exp=expectOf(d).trim();const g=got===exp;if(g)ok++;detail.push({exp,got,g});});
  console.log('  '+label+': '+ok+'/'+set.length);
  if(process.env.TAO_SHOW==='1') detail.forEach(x=>console.log('    '+(x.g?'✓':'✗')+' 期望='+JSON.stringify(x.exp)+' 模型='+JSON.stringify(x.got).slice(0,60)));
};

console.log('=== rebalanced_v2 评测  model='+DSB+' ===');
console.log('语料内样本: 原样复制='+verb.length+'  有序复制='+ord.length+'  算术='+ar.length+'  绑定='+bind.length);
console.log('-- 语料内 --');
runSet(pick(verb,12),'原样复制(语料内)',d=>d[0].slice(2),d=>d[1].slice(2));
runSet(pick(ord,12),'有序复制(语料内)',d=>d[0].slice(2),d=>d[1].slice(2));
runSet(pick(ar,12),'算术(语料内)',d=>d[0].slice(2),d=>d[1].slice(2));
