// 分类别评测：把 held-out union 集按能力类别打分。逐行解析，不用正则。
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
const PROMPTS='D:/TaoVm/data/noffn_probe2/heldout_union.txt';
const rules=[
 {n:1, cat:'A', d:'中国首都',   ok:r=>r.includes('北京')},
 {n:2, cat:'B', d:'1+1',        ok:r=>r.replace(/\s/g,'')==='2'},
 {n:3, cat:'A', d:'谢谢英文',   ok:r=>/thank/i.test(r)},
 {n:4, cat:'A', d:'自我介绍',   ok:r=>r.includes('夭夭')},
 {n:5, cat:'B', d:'3+2苹果',    ok:r=>r.replace(/\s/g,'')==='5'},
 {n:6, cat:'A', d:'法国首都',   ok:r=>r.includes('巴黎')},
 {n:7, cat:'C', d:'1+1换说法',  ok:r=>r.replace(/\s/g,'')==='2'},
 {n:8, cat:'B', d:'3+2',        ok:r=>r.replace(/\s/g,'')==='5'},
 {n:9, cat:'C', d:'首都换说法', ok:r=>r.includes('北京')},
 {n:10,cat:'A', d:'地球绕',     ok:r=>r.includes('太阳')},
 {n:11,cat:'A', d:'水的化学式', ok:r=>/H2O/i.test(r)},
 {n:12,cat:'A', d:'一年月数',   ok:r=>r.replace(/\s/g,'')==='12'},
 {n:13,cat:'A', d:'一周天数',   ok:r=>r.replace(/\s/g,'')==='7'},
 {n:14,cat:'C', d:'谢谢换说法', ok:r=>/thank/i.test(r)},
 {n:15,cat:'A', d:'朋友英文',   ok:r=>/friend/i.test(r)},
 {n:16,cat:'A', d:'你的名字',   ok:r=>r.includes('夭夭')},
 {n:17,cat:'D', d:'记住小何银色', ok:r=>r.includes('小何')&&r.includes('银色')},
 {n:18,cat:'D', d:'取回小何银色', ok:r=>r.includes('小何')&&r.includes('银色')},
 {n:19,cat:'E', d:'钢琴-承接',  ok:r=>/琴/.test(r)},
 {n:20,cat:'E', d:'钢琴-买什么',ok:r=>/琴/.test(r)},
 {n:21,cat:'E', d:'钢琴-练多久',ok:r=>/分钟|小时|半小时/.test(r)},
 {n:22,cat:'F', d:'顺序输出三色',ok:r=>{const i=r.indexOf('红'),j=r.indexOf('黄'),k=r.indexOf('蓝');return i>=0&&j>i&&k>j;}},
];
function parse(out){
  const lines=out.split(/\r?\n/); const res=[]; let cur=null;
  for(const ln of lines){
    if(ln.startsWith('BEGIN_REPLY ')){ cur={n:+ln.slice(12).trim(),text:[]}; continue; }
    if(ln.startsWith('END_REPLY ')){ if(cur){res.push({n:cur.n,text:cur.text.join('')});cur=null;} continue; }
    if(cur&&ln!=='RESET') cur.text.push(ln);
  }
  return res;
}
function run(dsb){
  const input=fs.readFileSync(PROMPTS,'utf8');
  const exe=process.env.TAO_CPU_EXE||'D:/TaoVm/build/yaoyao_cpu_v01.exe';
  const r=spawnSync(exe,[dsb],{cwd:'D:/TaoVm',input,encoding:'utf8',timeout:300000,maxBuffer:1<<26});
  return parse(String(r.stdout||''));
}
const summary=[];
for(const m of process.argv.slice(2)){
  const eq=m.indexOf('='); const label=m.slice(0,eq), dsb=m.slice(eq+1);
  const byN=Object.fromEntries(run(dsb).map(r=>[r.n,r.text]));
  const cats={}; let P=0,T=0; const lines=[];
  for(const rule of rules){
    const text=byN[rule.n]!==undefined?byN[rule.n]:'(missing)';
    const pass=byN[rule.n]!==undefined&&rule.ok(text);
    if(!cats[rule.cat])cats[rule.cat]={pass:0,total:0}; cats[rule.cat].total++; if(pass)cats[rule.cat].pass++;
    T++; if(pass)P++;
    lines.push(`  ${pass?'PASS':'FAIL'} [${rule.cat}] ${rule.d}  -> ${JSON.stringify(text).slice(0,60)}`);
  }
  console.log('\n########## '+label+' ##########');
  for(const l of lines) console.log(l);
  const cs=['A','B','C','D','E','F'].filter(c=>cats[c]).map(c=>`${c}:${cats[c].pass}/${cats[c].total}`).join('  ');
  console.log('  分类: '+cs); console.log('  TOTAL: '+P+'/'+T);
  summary.push({label,...cats,total:`${P}/${T}`});
}
console.log('\n\n===== 汇总 =====');
for(const s of summary) console.log(`${s.label}\tA:${s.A.pass}/${s.A.total}\tB:${s.B.pass}/${s.B.total}\tC:${s.C.pass}/${s.C.total}\tD:${s.D.pass}/${s.D.total}\tE:${s.E.pass}/${s.E.total}\tF:${s.F.pass}/${s.F.total}\t总:${s.total}`);