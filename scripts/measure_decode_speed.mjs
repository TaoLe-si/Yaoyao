
import {spawnSync} from 'node:child_process';
const DSB='D:/TaoVm/build/h2r_l6_long/step_500/final.dsb';
const POOL=['铅笔','草莓','桃子','桌子','绿色','钟表','鱼','北京','广州','兔子','紫色','马','水杯','杭州','白色','重庆','葡萄','橙子','红色','成都','椅子','黄色','地图','笔记本','橡皮','雨伞','猫','狗','蓝色','粉色','香蕉','苹果','西瓜','天津','武汉','书包','西安','鸡','尺子','橙色','上海','石头'];
let seed=4242;const rnd=()=>{seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff;};
const PROMPTS=[];
for(let i=0;i<600;i++){const L=2+Math.floor(rnd()*5);const it=[],u=new Set();
  while(it.length<L){const x=POOL[Math.floor(rnd()*POOL.length)];if(u.has(x))continue;u.add(x);it.push(x);}
  PROMPTS.push('请按顺序输出：'+it.join('、')+'。不要添加其他文字。');}
const inp=[];for(const p of PROMPTS)inp.push('/reset',p);inp.push('/quit');
const input=inp.join('\n')+'\n';
const env={...process.env,TAO_FAST_ACT:'0'};
const one=(exe)=>{const t0=Date.now();
  const r=spawnSync('D:/TaoVm/build/'+exe,[DSB],{cwd:'D:/TaoVm',input,encoding:'utf8',timeout:1800000,maxBuffer:1<<27,env});
  const dt=(Date.now()-t0)/1000;const o=String(r.stdout||'');
  const toks=[...o.matchAll(/END_REPLY tokens=(\d+)/g)].reduce((a,m)=>a+ +m[1],0);
  const reps=[...o.matchAll(/BEGIN_REPLY/g)].length;
  return {dt,toks,reps,err:String(r.stderr||'').slice(0,150)};};
one('h2r_cpu_base.exe');one('h2r_cpu_rev.exe');
const res={base:[],rev:[]};
for(let i=0;i<3;i++)for(const [k,exe] of [['base','h2r_cpu_base.exe'],['rev','h2r_cpu_rev.exe']])res[k].push(one(exe));
const med=a=>{const s=[...a].sort((x,y)=>x-y);return s[1];};
const out=[];
out.push('=== CPU 解码速度对比（600 条提示，同模型 h2r_l6_long/step_500，TAO_FAST_ACT=0）===');
for(const k of ['base','rev']){const R=res[k];
  out.push('  '+(k==='base'?'改动前 h2r_cpu_base':'回退后 h2r_cpu_rev ')+': '+R.map(x=>x.dt.toFixed(2)).join(' / ')+' s  中位 '+med(R.map(x=>x.dt)).toFixed(3)+' s  token '+R[0].toks+'  回复 '+R[0].reps+(R[0].err?'  ERR:'+R[0].err:''));}
const b=med(res['base'].map(x=>x.dt)),r=med(res['rev'].map(x=>x.dt));
out.push('');
out.push('  中位耗时比 (回退后/改动前) = '+(r/b).toFixed(4));
out.push('  判定: '+(r<=b*1.02?'✓ 无退化（≤2% 噪声带内）':'✗ 有退化'));
import fs from 'node:fs';
fs.writeFileSync('D:/TaoVm/build/decode_speed.txt',out.join('\n')+'\n');
console.log(out.join('\n'));
