import fs from 'node:fs';
const file=process.argv[2]||'build/filtered_repair_v1.log';
const text=fs.readFileSync(file,'utf8');
const rows=text.split(/\r?\n/).filter(l=>l.startsWith('UPDATE ')).map(l=>Object.fromEntries(l.slice(7).split(' ').map(x=>{const i=x.indexOf('=');return [x.slice(0,i), Number(x.slice(i+1))];})));
for(const [i,r] of rows.entries()){
  if(!Object.values(r).every(Number.isFinite)||r.step!==i+1||r.targets<=0||r.positions<r.targets) throw Error('invalid update '+JSON.stringify(r));
}
const n=rows.reduce((s,r)=>s+r.targets,0);
const tail=rows.slice(-10);
const initial=Number(text.match(/INITIAL .*?val_NLL=([0-9.e+-]+)/)?.[1]);
console.log(JSON.stringify({
  file, updates:rows.length,
  finished:/^FINAL step=200 updates=200 /m.test(text)&&rows.length===200,
  failed:text.includes('FILTERED_PILOT_FAIL'),
  initialValidation:initial,
  finalValidation:rows.at(-1)?.val_NLL??null,
  last10Mean:tail.length?tail.reduce((s,r)=>s+r.val_NLL,0)/tail.length:null,
  best:rows.length?rows.reduce((a,b)=>a.val_NLL<b.val_NLL?a:b):null,
  supervised:n,
  positions:rows.reduce((s,r)=>s+r.positions,0),
  weightedTrain:n?rows.reduce((s,r)=>s+r.targets*r.train_preupdate_NLL,0)/n:null,
  caveat:'Filtered corpus + Adam reset + lr .0001 from 1240 master. Not isolated causal test. Do not claim quality from NLL alone.'
},null,2));
