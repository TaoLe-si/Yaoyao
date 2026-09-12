// 读取训练日志：剥掉历史遗留的 UTF-16 报错前缀（混合编码），只输出 ASCII 正文。
// 用法: node scripts/logview.mjs <logfile> [outfile] [tailN]
//   - 省略 outfile 则直接打印；给 outfile 则写干净副本（可在编辑器/记事本正常打开）
//   - tailN 只输出最后 N 行正文（0/省略 = 全部）
import fs from 'node:fs';

const [inp, outArg, tailArg] = process.argv.slice(2);
if (!inp) { console.log('usage: node scripts/logview.mjs <logfile> [outfile] [tailN]'); process.exit(1); }
if (!fs.existsSync(inp)) { console.log('MISSING: ' + inp); process.exit(1); }

const buf = fs.readFileSync(inp);
const MARK = Buffer.from('CONFIG layers=', 'latin1');
const at = buf.indexOf(MARK);
let body;
if (at >= 0) body = buf.subarray(at).toString('utf8');
else if (buf.subarray(0, 64).includes(0)) body = buf.toString('utf16le').replace(/^\uFEFF/, '');
else body = buf.toString('utf8');

const lines = body.split(/\r?\n/);
const tailN = +(tailArg || 0);
const shown = tailN > 0 ? lines.slice(-tailN) : lines;
const text = shown.join('\n');

if (outArg) {
  fs.writeFileSync(outArg, body.trimEnd() + '\n');
  console.log('干净副本: ' + outArg + '  ' + lines.length + ' 行  ' + body.length + ' 字节');
  console.log(text.split('\n').filter(Boolean).slice(-4).join('\n'));
} else {
  console.log(text);
}
