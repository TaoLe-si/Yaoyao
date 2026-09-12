// 解码速度基准：分离"纯解码吞吐"与"预填充(ttft)"，并扫描线程数 × 惩罚口径。
// 用法: node scripts/bench_decode.mjs [dsb] [nPrompts] [reps]
// 输出 TSV 便于粘贴。注意：CPU 解码与 CUDA 训练并行时会互相争抢 CPU 线程。
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const DSB  = process.argv[2] || 'D:/TaoVm/build/s1_s8r1/final.dsb';
const N    = +(process.argv[3] || 200);
const REPS = +(process.argv[4] || 3);
const EXE  = process.env.TAO_CPU_EXE || 'D:/TaoVm/build/h2r_cpu.exe';
const TOK  = process.env.TAO_TOKENIZER || 'D:/TaoVm/build/tok_real_v1.bbp';

const POOL=['铅笔','草莓','桃子','桌子','绿色','钟表','鱼','北京','广州','兔子','紫色','马','水杯','杭州','白色','重庆','葡萄','橙子','红色','成都','椅子','黄色','地图','笔记本','橡皮','雨伞','猫','狗','蓝色','粉色','香蕉','苹果','西瓜','天津','武汉','书包','西安','鸡','尺子','橙色','上海','石头'];

// A 类：短指令，输出短列表 -> 解码主导
function mkShort(i, rnd){
  const L=2+Math.floor(rnd()*5); const it=[],u=new Set();
  while(it.length<L){const x=POOL[Math.floor(rnd()*POOL.length)];if(u.has(x))continue;u.add(x);it.push(x);}
  return '请按顺序输出：'+it.join('、')+'。不要添加其他文字。';
}
// B 类：长上下文，短回答 -> 预填充主导
function mkLong(i, rnd){
  const parts=[]; for(let k=0;k<60;k++) parts.push(POOL[Math.floor(rnd()*POOL.length)]);
  return '下面是一段资料：'+parts.join('，')+'。请只回答一个问题：第二个词是什么？';
}
const mk = (cls,i,rnd) => cls==='A' ? mkShort(i,rnd) : mkLong(i,rnd);

function buildInput(cls){
  let seed=4242; const rnd=()=>{seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff;};
  const out=[];
  for(let i=0;i<N;i++){ out.push('/reset', mk(cls,i,rnd)); }
  out.push('/quit');
  return out.join('\n')+'\n';
}

function run(cls, threads, pen){
  const input=buildInput(cls);
  const t0=Date.now();
  const r=spawnSync(EXE,[DSB],{cwd:'D:/TaoVm',input,encoding:'utf8',timeout:3600000,maxBuffer:1<<27,
    env:{...process.env,TAO_TOKENIZER:TOK,TAO_CPU_THREADS:String(threads),TAO_REP_PEN:String(pen),TAO_REP_WIN:'0'}});
  const wall=(Date.now()-t0)/1000;
  const o=String(r.stdout||'');
  const toks=[...o.matchAll(/END_REPLY tokens=(\d+)/g)].map(m=>+m[1]);
  const tps =[...o.matchAll(/decode_tps=([\d.]+)/g)].map(m=>+m[1]).filter(Number.isFinite);
  const ttft=[...o.matchAll(/ttft_compute=([\d.eE+-]+)/g)].map(m=>+m[1]).filter(Number.isFinite);
  const trun=[...o.matchAll(/truncated=(\d+)/g)].map(m=>+m[1]).reduce((a,b)=>a+b,0);
  const empty=(o.match(/\[empty reply\]/g)||[]).length;
  const sorted=[...tps].sort((a,b)=>a-b);
  const med=sorted.length?sorted[Math.floor(sorted.length/2)]:0;
  const sumT=toks.reduce((a,b)=>a+b,0);
  return {wall,reps:toks.length,sumT,empty,trun,
    meanTps:tps.length?tps.reduce((a,b)=>a+b,0)/tps.length:0, medTps:med,
    e2e:wall>0?sumT/wall:0, ttft:ttft.length?ttft.reduce((a,b)=>a+b,0)/ttft.length:0,
    decodeSec:tps.reduce((a,t,i)=>a+toks[i]/t,0), code:r.status, err:String(r.stderr||'').slice(0,160)};
}

if(!fs.existsSync(DSB)){console.log('MISSING dsb: '+DSB);process.exit(1);}
if(!fs.existsSync(EXE)){console.log('MISSING exe: '+EXE);process.exit(1);}

console.log('=== 解码速度基准 ===');
console.log('model : '+DSB);
console.log('exe   : '+EXE+'   threads 扫描, prompts='+N+', 每配置 '+REPS+' 次取中位');
console.log('');
console.log(['类','线程','pen','中位解码tps','均解码tps','端到端tps','纯解码秒','总秒','ttft均(ms)','回复','token','空','截断'].join('\t'));
for(const cls of ['A','B']){
  const threadList = cls==='A' ? [1,2,4,8,16] : [8];
  for(const th of threadList){
    const pens = cls==='A' ? [1.3,0] : [1.3];
    for(const pen of pens){
      const rs=[]; for(let k=0;k<REPS;k++) rs.push(run(cls,th,pen));
      const med=a=>{const s=[...a].sort((x,y)=>x-y);return s[Math.floor(s.length/2)];};
      const f=x=>x.toFixed(1);
      console.log([cls,th,pen,f(med(rs.map(r=>r.medTps))),f(med(rs.map(r=>r.meanTps))),
        f(med(rs.map(r=>r.e2e))),med(rs.map(r=>r.decodeSec)).toFixed(3),med(rs.map(r=>r.wall)).toFixed(3),
        (med(rs.map(r=>r.ttft))*1000).toFixed(2),med(rs.map(r=>r.reps)),med(rs.map(r=>r.sumT)),
        rs[0].empty,rs[0].trun].join('\t'));
      const bad=rs.find(r=>r.code!==0); if(bad) console.log('  NONZERO_EXIT code='+bad.code);
    }
  }
}
console.log('DONE');
