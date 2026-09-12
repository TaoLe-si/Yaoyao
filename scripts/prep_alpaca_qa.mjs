// alpaca_zh.json（instruction/input/output 数组）→ 问答训练 jsonl + held-out + BPE 语料。
// 用法: node scripts/prep_alpaca_qa.mjs
import fs from 'node:fs';
import path from 'node:path';

const SRC = 'E:/taovm-data/alpaca_zh.json';
const OUT = 'D:/TaoVm/data/alpaca_qa';
const HOLD = 200;
const SEED0 = 20260912;

function userOf(r) {
  const ins = String(r.instruction || '').trim();
  const inp = String(r.input || '').trim();
  if (!ins) return '';
  return inp ? ins + '\n' + inp : ins;
}

if (!fs.existsSync(SRC)) {
  console.error('ALPACA_QA_FAIL missing ' + SRC);
  process.exit(1);
}
fs.mkdirSync(OUT, { recursive: true });
for (const name of ['train.jsonl', 'heldout.jsonl', 'tok_text.txt']) {
  const p = path.join(OUT, name);
  if (fs.existsSync(p)) {
    console.error('ALPACA_QA_FAIL refuse overwrite ' + p);
    process.exit(1);
  }
}

const rows = JSON.parse(fs.readFileSync(SRC, 'utf8'));
if (!Array.isArray(rows) || rows.length < HOLD + 100) {
  console.error('ALPACA_QA_FAIL bad source n=' + (rows && rows.length));
  process.exit(1);
}

const items = [];
for (const r of rows) {
  const user = userOf(r);
  const ref = String(r.output || '').trim();
  if (!user || !ref) continue;
  items.push({ user, ref });
}

let seed = SEED0 >>> 0;
const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
const idx = items.map((_, i) => i);
for (let i = idx.length - 1; i > 0; --i) {
  const j = Math.floor(rnd() * (i + 1));
  const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
}
const holdSet = new Set(idx.slice(0, HOLD));

const train = fs.createWriteStream(path.join(OUT, 'train.jsonl'), { encoding: 'utf8' });
const held = fs.createWriteStream(path.join(OUT, 'heldout.jsonl'), { encoding: 'utf8' });
const tok = fs.createWriteStream(path.join(OUT, 'tok_text.txt'), { encoding: 'utf8' });
let nTrain = 0, nHold = 0, tokBytes = 0;
const esc = (s) => JSON.stringify(s);
for (let i = 0; i < items.length; ++i) {
  const { user, ref } = items[i];
  if (holdSet.has(i)) {
    held.write('{"user":' + esc(user) + ',"ref":' + esc(ref) + '}\n');
    nHold++;
    continue;
  }
  train.write('{"prompt":' + esc(user) + ',"response":' + esc(ref) + '}\n');
  tok.write(user.replace(/\s+/g, ' ') + '\n' + ref.replace(/\s+/g, ' ') + '\n');
  tokBytes += Buffer.byteLength(user) + Buffer.byteLength(ref) + 2;
  nTrain++;
}
await Promise.all([
  new Promise((r, e) => train.end((err) => err ? e(err) : r())),
  new Promise((r, e) => held.end((err) => err ? e(err) : r())),
  new Promise((r, e) => tok.end((err) => err ? e(err) : r())),
]);
console.log('ALPACA_QA_DONE source=' + items.length + ' train=' + nTrain + ' heldout=' + nHold
  + ' tok_bytes=' + tokBytes + ' out=' + OUT);
if (nHold !== HOLD) {
  console.error('ALPACA_QA_FAIL heldout=' + nHold);
  process.exit(1);
}
