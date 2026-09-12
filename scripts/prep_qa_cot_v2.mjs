// v2 mix: alpaca-majority QA + shorter APE CoT + digit CoT with multiple ask forms.
// Fixes v1 failure mode: APE "思考：计算…答案" became the default reply for every prompt.
// Usage: node scripts/prep_qa_cot_v2.mjs
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

const OUT = 'D:/TaoVm/data/qa_cot_v2';
const APE = 'E:/taovm-data/real_reason/ape210k_dialog.txt';
const ALPACA = 'D:/TaoVm/data/alpaca_qa/train.jsonl';
const N_APE = 8000;
const N_ALPACA = 36000;
const N_APE_HOLD = 200;
const APE_ANS_MAX = 360;
const SEED0 = 20260913;

function rndMake(seed0) {
  let seed = seed0 >>> 0;
  return () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
}
function shuffle(a, rnd) {
  for (let i = a.length - 1; i > 0; --i) {
    const j = Math.floor(rnd() * (i + 1));
    const t = a[i]; a[i] = a[j]; a[j] = t;
  }
  return a;
}
function esc(s) { return JSON.stringify(s); }

if (fs.existsSync(path.join(OUT, 'train.jsonl'))) {
  console.error('QA_COT_V2_FAIL refuse overwrite ' + OUT);
  process.exit(1);
}
if (!fs.existsSync(APE) || !fs.existsSync(ALPACA)) {
  console.error('QA_COT_V2_FAIL missing APE or alpaca jsonl');
  process.exit(1);
}
fs.mkdirSync(OUT, { recursive: true });

async function loadApe() {
  const rl = readline.createInterface({ input: fs.createReadStream(APE, { encoding: 'utf8' }), crlfDelay: Infinity });
  const docs = [];
  let u = '', a = '';
  const flush = () => {
    const user = u.trim(), ans = a.trim();
    u = ''; a = '';
    if (!user || !ans) return;
    if (ans.length < 20 || ans.length > APE_ANS_MAX) return;
    if (!ans.includes('计算') || !ans.includes('答案')) return;
    const m = ans.match(/答案[：:]\s*(.+)$/);
    if (!m || !String(m[1]).trim()) return;
    const body = ans.startsWith('思考') ? ans : ('思考：' + ans);
    docs.push({ user, ref: body });
  };
  for await (const raw of rl) {
    let line = raw;
    if (line.endsWith('\r')) line = line.slice(0, -1);
    if (line === 'DOC') { flush(); continue; }
    if (line.startsWith('U ')) u = line.slice(2);
    else if (line.startsWith('A ')) a = line.slice(2);
  }
  flush();
  return docs;
}

function askPlus(a, b, k) {
  const s = `${a}+${b}`;
  switch (k % 5) {
    case 0: return `${s}等于多少？请先思考再作答。`;
    case 1: return `${s}等于几？`;
    case 2: return `${a}加${b}等于多少？`;
    case 3: return `请计算${s}`;
    default: return `${s}等于多少？请写出思考过程和答案。`;
  }
}
function askMinus(a, b, k) {
  const s = `${a}-${b}`;
  switch (k % 5) {
    case 0: return `${s}等于多少？请先思考再作答。`;
    case 1: return `${s}等于几？`;
    case 2: return `${a}减${b}等于多少？`;
    case 3: return `请计算${s}`;
    default: return `${s}等于多少？请写出思考过程和答案。`;
  }
}

function digitCot(rnd) {
  const seen = [];
  for (let a = 1; a <= 20; a++) for (let b = 1; b <= 20; b++) seen.push([a, b]);
  shuffle(seen, rnd);
  const trainPairs = seen.slice(0, 320);
  const holdPairs = seen.slice(320);
  const docs = [];
  const hold = [];
  const chainPlus = (a, b) => {
    const s = a + b;
    const ones = (a % 10) + (b % 10);
    const carry = ones >= 10 ? 1 : 0;
    const write = ones >= 10 ? ones - 10 : ones;
    const tens = Math.floor(a / 10) + Math.floor(b / 10) + carry;
    return {
      think: `思考：个位 ${a % 10} 加 ${b % 10} 得 ${ones}，写 ${write} 进 ${carry}；十位 ${Math.floor(a / 10)} 加 ${Math.floor(b / 10)} 加进位 ${carry} 得 ${tens}。\n答案：${s}`,
      ans: String(s),
    };
  };
  const chainMinus = (a, b) => {
    if (a < b) return null;
    const d = a - b;
    const a1 = a % 10, b1 = b % 10;
    const borrow = a1 < b1 ? 1 : 0;
    const ones = borrow ? a1 + 10 - b1 : a1 - b1;
    const tens = Math.floor(a / 10) - Math.floor(b / 10) - borrow;
    return {
      think: `思考：个位 ${a1} 减 ${b1}${borrow ? '不够减，向十位借 1 变成 ' + (a1 + 10) + '，' + (a1 + 10) + ' 减 ' + b1 + ' 得 ' + ones : '得 ' + ones}；十位 ${Math.floor(a / 10)} 减 ${Math.floor(b / 10)}${borrow ? '再减借位 1' : ''} 得 ${tens}。\n答案：${d}`,
      ans: String(d),
    };
  };
  for (let k = 0; k < 5; k++) {
    for (const [a, b] of trainPairs) {
      const p = chainPlus(a, b);
      docs.push({ user: askPlus(a, b, k), ref: p.think });
      const m = chainMinus(a, b);
      if (m) docs.push({ user: askMinus(a, b, k), ref: m.think });
    }
  }
  for (const [a, b] of holdPairs.slice(0, 80)) {
    const p = chainPlus(a, b);
    hold.push({ user: askPlus(a, b, 1), ref: p.ans, kind: 'digit+' });
  }
  return { docs, hold };
}

const rnd = rndMake(SEED0);
const ape = shuffle(await loadApe(), rnd);
if (ape.length < N_APE + N_APE_HOLD) {
  console.error('QA_COT_V2_FAIL ape kept=' + ape.length);
  process.exit(1);
}
const apeHold = ape.slice(0, N_APE_HOLD).map((d) => {
  const m = String(d.ref).match(/答案[：:]\s*(.+)$/);
  return { user: d.user, ref: m ? m[1].trim() : d.ref, kind: 'ape' };
});
const apeTrain = ape.slice(N_APE_HOLD, N_APE_HOLD + N_APE);

const alpacaLines = fs.readFileSync(ALPACA, 'utf8').split('\n').filter(Boolean);
shuffle(alpacaLines, rnd);
if (alpacaLines.length < N_ALPACA) {
  console.error('QA_COT_V2_FAIL alpaca=' + alpacaLines.length);
  process.exit(1);
}
const alpaca = alpacaLines.slice(0, N_ALPACA).map((l) => JSON.parse(l));

const digit = digitCot(rndMake(SEED0 + 7));
const train = shuffle([
  ...apeTrain.map((d) => ({ prompt: d.user, response: d.ref })),
  ...alpaca.map((d) => ({ prompt: d.prompt, response: d.response })),
  ...digit.docs.map((d) => ({ prompt: d.user, response: d.ref })),
], rnd);

const tw = fs.createWriteStream(path.join(OUT, 'train.jsonl'), { encoding: 'utf8' });
for (const d of train) tw.write('{"prompt":' + esc(d.prompt) + ',"response":' + esc(d.response) + '}\n');
await new Promise((r, e) => tw.end((err) => err ? e(err) : r()));

const held = [...apeHold, ...digit.hold];
fs.writeFileSync(path.join(OUT, 'heldout_cot.jsonl'), held.map((d) => JSON.stringify(d)).join('\n') + '\n');
fs.writeFileSync(path.join(OUT, 'heldout_qa.jsonl'),
  fs.readFileSync('D:/TaoVm/data/alpaca_qa/heldout.jsonl'));

console.log('QA_COT_V2_DONE train=' + train.length
  + ' ape=' + apeTrain.length + '/' + ape.length
  + ' alpaca=' + alpaca.length
  + ' digit=' + digit.docs.length
  + ' held_cot=' + held.length
  + ' out=' + OUT);
