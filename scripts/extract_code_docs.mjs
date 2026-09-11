// 流式解压 codeparrot json.gz -> 标记分隔的纯文本（一文件一档）。
// 不能用 gunzipSync：解压后 >512MB，超出 Node 字符串上限。
import fs from 'node:fs';
import zlib from 'node:zlib';
import readline from 'node:readline';

const SRC = process.argv[2];
const OUT = process.argv[3];
const MIN = Number(process.argv[4] || 64);
const MAX = Number(process.argv[5] || 200000);

const inp = fs.createReadStream(SRC).pipe(zlib.createGunzip());
const out = fs.createWriteStream(OUT);
const rl = readline.createInterface({ input: inp, crlfDelay: Infinity });
let n = 0, bytes = 0, skipped = 0;
rl.on('line', l => {
  if (!l.trim()) return;
  let j; try { j = JSON.parse(l); } catch { return; }
  const c = j.content;
  if (typeof c !== 'string') return;
  if (c.length < MIN || c.length > MAX) { skipped++; return; }
  out.write(c); out.write('\n@@DOC@@\n');
  n++; bytes += c.length;
});
rl.on('close', () => out.end(() => {
  console.log('EXTRACT_DONE docs=' + n + ' bytes=' + bytes + ' skipped=' + skipped);
}));
