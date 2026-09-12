// 基座模型文本续写演示（GPT-2 式）：给前缀，看它续写。
// 用法: node scripts/show_continuation.mjs <model.dsb> [pen] [n]
const MAXTOK = Number(process.env.TAO_MAX_TOKENS || 256);
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
const MODEL=process.argv[2]||'build/n3_c/final.dsb';
const PEN=process.argv[3]||'1.15';
const N=Number(process.argv[4]||12);
const PREFIXES=[
 '昭通机场位于中国云南省昭通市，',
 '机器学习是人工智能的一个分支，',
 '在量子力学中，波函数描述的是',
 '中国古代的科举制度始于隋朝，',
 '太阳系由太阳和围绕它运动的',
 '这篇文章介绍了如何使用Python编写一个',
 '唐朝是中国历史上最强盛的朝代之一，',
 '水在标准大气压下的沸点是',
 '第二次世界大战结束后，世界格局',
 '计算机程序的执行过程可以分为',
 '银河系是一个包含数千亿颗恒星的',
 '光合作用是绿色植物利用光能',
 '文艺复兴时期的艺术家们',
 '经济学中的供需关系指的是',
 '人类基因组计划是一项',
 '地震是由地壳运动引起的',
];
const pick=PREFIXES.slice(0,N);
const input=[];
for(const p of pick){ input.push('/reset'); input.push(p); }
input.push('/quit');
const t0=Date.now();
const p=spawnSync('D:/TaoVm/build/h2r_cpu.exe',[MODEL,'--rep-pen',PEN,'--rep-win','0','--max',String(MAXTOK)],{
 cwd:'D:/TaoVm',input:input.join('\n')+'\n',encoding:'utf8',timeout:1800000,maxBuffer:1<<28,
 env:{...process.env,TAO_TOKENIZER:'D:/TaoVm/build/tok_real_v1.bbp',TAO_CPU_THREADS:'8',TAO_REP_PEN:PEN}});
const bodies=[...String(p.stdout||'').matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map(m=>m[1]);
console.log('==== '+MODEL+'  rep_pen='+PEN+'  用时'+((Date.now()-t0)/1000).toFixed(0)+'s ====');
let totLen=0,deg=0;
pick.forEach((pre,i)=>{
  const b=(bodies[i]||'').replace(/\s+$/,'');
  totLen+=b.length;
  // 退化检测：同一 8+ 字片段出现 3 次以上
  const seen={}; let isDeg=false;
  for(let k=0;k+8<=b.length;k++){ const g=b.slice(k,k+8); seen[g]=(seen[g]||0)+1; if(seen[g]>=3){isDeg=true;break;} }
  if(isDeg)deg++;
  console.log('\n['+(i+1)+'] 前缀: '+pre);
  console.log('    续写: '+(b||'(空)').slice(0,220));
});
console.log('\n---- 汇总: '+pick.length+' 条, 平均续写长度 '+(totLen/pick.length).toFixed(0)+' 字, 退化重复 '+deg+'/'+pick.length+' ('+(deg/pick.length*100).toFixed(1)+'%) ----');
