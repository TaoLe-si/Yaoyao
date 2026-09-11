
// CPU decode speed for the CURRENT architecture (H2R dual-state-4, Pade activation).
// Usage: node scripts/measure_decode_speed_v2.mjs [dsb] [nPrompts]
//
// NOTE: do NOT set TAO_FAST_ACT=0. The trainer now uses the decoder's Pade
// activation (TAO_TRAIN_FAST_ACT default ON), so the decoder must use its
// default fast_act=1. Forcing exact activation here would re-introduce the R4
// train/infer mismatch that caused the degenerate-repetition defect.
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const DSB = process.argv[2] || 'D:/TaoVm/build/rebal_v2/final.dsb';
const N   = +(process.argv[3] || 600);
const EXE = process.env.TAO_CPU_EXE || 'D:/TaoVm/build/h2r_cpu.exe';

const POOL=['铅笔','草莓','桃子','桌子','绿色','钟表','鱼','北京','广州','兔子','紫色','马','水杯','杭州','白色','重庆','葡萄','橙子','红色','成都','椅子','黄色','地图','笔记本','橡皮','雨伞','猫','狗','蓝色','粉色','香蕉','苹果','西瓜','天津','武汉','书包','西安','鸡','尺子','橙色','上海','石头'];
let seed=4242;const rnd=()=>{seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff;};
const PROMPTS=[];
for(let i=0;i<N;i++){
  const L=2+Math.floor(rnd()*5);const it=[],u=new Set();
  while(it.length<L){const x=POOL[Math.floor(rnd()*POOL.length)];if(u.has(x))continue;u.add(x);it.push(x);}
  PROMPTS.push('请按顺序输出：'+it.join('、')+'。不要添加其他文字。');
}
const inp=[];for(const p of PROMPTS)inp.push('/reset',p);inp.push('/quit');
const input=inp.join('\n')+'\n';

function once(){
  const t0=Date.now();
  const r=spawnSync(EXE,[DSB],{cwd:'D:/TaoVm',input,encoding:'utf8',timeout:1800000,maxBuffer:1<<27});
  const wall=(Date.now()-t0)/1000;
  const o=String(r.stdout||'');
  const toks=[...o.matchAll(/END_REPLY tokens=(\d+)/g)].reduce((a,m)=>a+ +m[1],0);
  const reps=[...o.matchAll(/BEGIN_REPLY/g)].length;
  const tps=[...o.matchAll(/decode_tps=([\d.]+)/g)].map(m=>+m[1]).filter(x=>Number.isFinite(x));
  const sumTps=tps.reduce((a,b)=>a+b,0);
  const empty=[...o.matchAll(/\[empty reply\]/g)].length;
  return {wall,toks,reps,meanTps:tps.length?sumTps/tps.length:0,empty,err:String(r.stderr||'').slice(0,200)};
}

if(!fs.existsSync(DSB)){ console.log('MISSING dsb: '+DSB); process.exit(1); }
if(!fs.existsSync(EXE)){ console.log('MISSING exe: '+EXE); process.exit(1); }

const runs=[];
for(let i=0;i<3;i++) runs.push(once());
const med=a=>{const s=[...a].sort((x,y)=>x-y);return s[Math.floor(s.length/2)];};
const lines=[];
lines.push('=== CPU 解码速度（当前架构 dual-state-4-noffn-delta-mem-input-sqrt-d，Padé 默认）===');
lines.push('  model : '+DSB);
lines.push('  exe   : '+EXE);
lines.push('  prompts='+N+'  3 次交替取中位');
for(const [i,r] of runs.entries())
  lines.push('  run'+(i+1)+': wall='+r.wall.toFixed(2)+'s  tokens='+r.toks+'  replies='+r.reps+
             '  mean_decode_tps='+r.meanTps.toFixed(1)+'  empty='+r.empty+(r.err?'  ERR:'+r.err:''));
const mw=med(runs.map(r=>r.wall)), mt=med(runs.map(r=>r.meanTps)), mk=med(runs.map(r=>r.toks));
lines.push('');
lines.push('  中位 wall='+mw.toFixed(3)+'s  中位 tokens='+mk);
lines.push('  ==> 端到端吞吐 = '+(mk/mw).toFixed(1)+' token/s（含预填充与进程启动）');
lines.push('  ==> 纯解码吞吐 = '+mt.toFixed(1)+' token/s（decoder 自报 decode_tps 中位）');
fs.writeFileSync('D:/TaoVm/build/decode_speed_v2.txt',lines.join('\n')+'\n');
console.log(lines.join('\n'));
