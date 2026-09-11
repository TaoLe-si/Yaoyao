// 流式抽取 wiki JSON 的 completion 字段 -> @@DOC@@ 分隔纯文本
// 500MB 超过 Node 单字符串上限，必须 createReadStream 流式处理
import fs from 'node:fs';
import readline from 'node:readline';

const OUT = 'E:/taovm-data/wiki_docs.txt';
const w = fs.createWriteStream(OUT, { encoding: 'utf8' });
const rl = readline.createInterface({ input: fs.createReadStream('E:/taovm-data/wiki_cn.json', { encoding: 'utf8' }), crlfDelay: Infinity });
let n = 0, bytes = 0;
for await (const line of rl) {
  const t = line.trim();
  if (!t.startsWith('"completion"')) continue;
  let v;
  try { v = JSON.parse('{' + t.replace(/,\s*$/, '') + '}').completion; } catch (e) { console.error('parse fail @' + n); continue; }
  if (!v || v.length < 40) continue;
  const doc = v.replace(/\r/g, '');
  w.write(doc + '\n@@DOC@@\n');
  n++; bytes += Buffer.byteLength(doc);
}
await new Promise(r => w.end(r));
console.log('docs=' + n + ' textBytes=' + bytes + ' (' + (bytes/1048576).toFixed(1) + 'MB)');
