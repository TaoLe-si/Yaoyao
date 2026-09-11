
import fs from 'node:fs';
import readline from 'node:readline';
const rl = readline.createInterface({ input: fs.createReadStream('E:/taovm-data/wiki_cn.json',{encoding:'utf8'}), crlfDelay: Infinity });
let i=0, hits=0;
for await (const line of rl) {
  const t=line.trim();
  if(i<5) console.log('LINE'+i+': '+JSON.stringify(t.slice(0,40)));
  if(t.startsWith('"completion"')) hits++;
  if(++i>=20000) break;
}
console.log('scanned='+i+' completionHits='+hits);
