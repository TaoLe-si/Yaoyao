// 配对 bootstrap + McNemar：判断两个检查点的差异是否显著。
// 用法: node scripts/paired_bootstrap.mjs <A.persample.jsonl> <B.persample.jsonl> [迭代数]
// 解释: A - B。配对比较消掉"题目难度"这一共同方差，比独立比较灵敏得多。
import fs from 'node:fs';
const A = JSON.parse('[' + fs.readFileSync(process.argv[2],'utf8').trim().split('\n').join(',') + ']');
const B = JSON.parse('[' + fs.readFileSync(process.argv[3],'utf8').trim().split('\n').join(',') + ']');
const ITER = Number(process.argv[4] || 10000);
const n = Math.min(A.length, B.length);
if (A.length !== B.length) console.log('[警告] 长度不同 A=' + A.length + ' B=' + B.length + '，只用前 ' + n + ' 条');
for (let i = 0; i < n; i++) if (A[i].i !== B[i].i) { console.log('[错误] 第 ' + i + ' 条索引不对齐: ' + A[i].i + ' vs ' + B[i].i); process.exit(1); }
console.log('配对样本 n=' + n + '  bootstrap 迭代=' + ITER + '\n');

const mean = v => v.reduce((s,x)=>s+x,0)/v.length;
// 简单确定性 RNG（xorshift），保证可复现
let seed = 20260912;
const rnd = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 4294967296; };

const CONT = [['f1','bigramF1'], ['lenRatio','长度比'], ['maxRun','最长单字重复'], ['gotLen','输出字符数']];
const BIN  = [['empty','空回复'], ['garbled','乱码'], ['loop','退化复读'], ['charLoop','单字循环']];

console.log('指标             A均值     B均值     差值(A-B)    95%置信区间            显著?');
console.log('---------------+--------+---------+-----------+----------------------+------');
for (const [k,label] of CONT) {
  const a = A.slice(0,n).map(o=>o[k]), b = B.slice(0,n).map(o=>o[k]);
  const d = a.map((x,i)=>x-b[i]); const obs = mean(d);
  const boots = new Array(ITER);
  for (let t=0;t<ITER;t++){ let s=0; for(let i=0;i<n;i++) s+=d[(rnd()*n)|0]; boots[t]=s/n; }
  boots.sort((x,y)=>x-y);
  const lo=boots[(ITER*0.025)|0], hi=boots[(ITER*0.975)|0];
  const sig = (lo>0||hi<0) ? '是' : '否';
  console.log(label.padEnd(14)+' '+mean(a).toFixed(4).padStart(7)+' '+mean(b).toFixed(4).padStart(8)+' '
    + (obs>=0?'+':'')+obs.toFixed(4).padStart(9)+'   ['+lo.toFixed(4)+', '+hi.toFixed(4)+']'
    + ' '.repeat(Math.max(0,7-lo.toFixed(4).length-hi.toFixed(4).length)) + (sig==='是'?'是 ★':'否'));
}
console.log('\n配对二元指标 (McNemar 精确检验)');
console.log('指标        A=1 B=0   A=0 B=1   净变化     二项p值    显著?');
console.log('----------+---------+---------+----------+----------+------');
for (const [k,label] of BIN) {
  let a1b0=0,a0b1=0;
  for (let i=0;i<n;i++){ const x=A[i][k]?1:0,y=B[i][k]?1:0; if(x===1&&y===0)a1b0++; if(x===0&&y===1)a0b1++; }
  const m=a1b0+a0b1;
  let p=1;
  if (m>0){ // 精确二项检验 p=0.5, 双侧（对数空间 + log-sum-exp，避免下溢）
    const logFact=new Float64Array(m+1);
    for(let i=1;i<=m;i++) logFact[i]=logFact[i-1]+Math.log(i);
    const kmin=Math.min(a1b0,a0b1);
    let mx=-Infinity; const terms=[];
    for(let k2=0;k2<=kmin;k2++){ const t=logFact[m]-logFact[k2]-logFact[m-k2]-m*Math.LN2; terms.push(t); if(t>mx)mx=t; }
    let acc=0; for(const t of terms) acc+=Math.exp(t-mx);
    p=Math.min(1,2*Math.exp(mx)*acc);
  }
  const delta=mean(A.slice(0,n).map(o=>o[k]))-mean(B.slice(0,n).map(o=>o[k]));
  console.log(label.padEnd(9)+' '+String(a1b0).padStart(7)+' '+String(a0b1).padStart(9)+' '
    +(delta>=0?'+':'')+delta.toFixed(4).padStart(8)+'  '+p.toFixed(4).padStart(8)+'  '+(p<0.05?'是 ★':'否'));
}
