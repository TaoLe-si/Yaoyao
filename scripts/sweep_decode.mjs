// 解码参数扫描：同一组前缀，比较退化重复率
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
const MODEL=process.argv[2]||'build/s2_night1/step_50659/final.dsb';
const PRE=[
 '昭通机场位于中国云南省昭通市，','机器学习是人工智能的一个分支，','在量子力学中，波函数描述的是',
 '中国古代的科举制度始于隋朝，','太阳系由太阳和围绕它运动的','这篇文章介绍了如何使用Python编写一个',
 '唐朝是中国历史上最强盛的朝代之一，','水在标准大气压下的沸点是','第二次世界大战结束后，世界格局',
 '计算机程序的执行过程可以分为','银河系是一个包含数千亿颗恒星的','光合作用是绿色植物利用光能',
 '文艺复兴时期的艺术家们','经济学中的供需关系指的是','人类基因组计划是一项','地震是由地壳运动引起的'];
const deg=(b)=>{const s={};for(let k=0;k+8<=b.length;k++){const g=b.slice(k,k+8);s[g]=(s[g]||0)+1;if(s[g]>=3)return true;}return false;};
const cfgs=[[1.0,8],[1.15,8],[1.3,8],[1.5,8],[1.3,0],[1.5,0],[1.2,16],[1.4,4]];
console.log('模型: '+MODEL);
console.log('pen   win   退化率      平均长度   样例片段');
for(const [pen,win] of cfgs){
  const input=[];for(const p of PRE){input.push('/reset');input.push(p);}
  input.push('/quit');
  const r=spawnSync('D:/TaoVm/build/h2r_cpu.exe',[MODEL,'--rep-pen',String(pen),'--rep-win',String(win)],{
    cwd:'D:/TaoVm',input:input.join('\n')+'\n',encoding:'utf8',timeout:900000,maxBuffer:1<<28,
    env:{...process.env,TAO_TOKENIZER:'D:/TaoVm/build/tok_real_v1.bbp',TAO_CPU_THREADS:'6'}});
  const b=[...String(r.stdout||'').matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map(m=>m[1]);
  const d=b.filter(deg).length; const avg=b.reduce((s,x)=>s+x.length,0)/Math.max(1,b.length);
  console.log(String(pen).padEnd(5)+String(win).padEnd(6)+String((d/b.length*100).toFixed(1)+'%').padEnd(11)+String(avg.toFixed(0)).padEnd(10)+(b[2]||'').slice(0,60).replace(/\n/g,' '));
}
console.log('SWEEP_DONE');
