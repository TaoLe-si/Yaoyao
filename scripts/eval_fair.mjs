// 按语料**真实格式**评测四项任务（此前的探针是分布外的，会低估模型）。
// 用法: node scripts/eval_fair.mjs <model.dsb> [tokenizer.bbp]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';

const DSB = process.argv[2];
const TOK = process.argv[3] || 'D:/TaoVm/build/tok_digit_v1.bbp';
const SENT = fs.readFileSync('D:/TaoVm/data/verbatim_diverse/sentences.txt', 'utf8')
  .split('\n').map(s => s.trim()).filter(Boolean);

// 从语料里抽取真实模板实例
const corpus = fs.readFileSync('D:/TaoVm/data/rebalanced_v2/conversations.txt', 'utf8').split('\n');
const orderLines = corpus.filter(l => l.startsWith('U 请按顺序输出：'));
const nameLines  = corpus.filter(l => l.startsWith('U 我的名字叫'));
const arithLines = corpus.filter(l => l.startsWith('U 只输出数字：'));

const pick = (arr, n, seed) => { let s = seed; const out = [];
  for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; out.push(arr[s % arr.length]); }
  return out; };

// 构造会话：每项任务一组 (prompt, expected)，同一会话内按序发送
const sessions = [];

// 1) 原样复制（语料内句子）
for (const s of pick(SENT, 12, 7)) {
  sessions.push({ task: 'verbatim', turns: [['请重复这句话：' + s, s]] });
}
// 2) 有序复制（语料真实行）
for (const u of pick(orderLines, 12, 11)) {
  sessions.push({ task: 'order', turns: [[u.slice(2), u.slice(2).replace(/^请按顺序输出：/, '').replace(/。不要添加其他文字。$/, '')]] });
}
// 3) 名字绑定（两轮，必须同会话）
for (const u of pick(nameLines, 12, 13)) {
  const body = u.slice(2);
  const name = (body.match(/我的名字叫(.+?)，/) || [])[1] || '';
  sessions.push({ task: 'binding', turns: [[body, '记住了。'], ['我叫什么名字？', name]] });
}
// 4) 算术（语料真实格式，含 × 与 +）
for (const u of pick(arithLines, 12, 17)) {
  const body = u.slice(2);
  const m = body.match(/只输出数字：(\d+)([×+])(\d+)等于多少？/);
  const exp = m ? String(m[2] === '×' ? Number(m[1]) * Number(m[3]) : Number(m[1]) + Number(m[3])) : '?';
  sessions.push({ task: 'arith', turns: [[body, exp]] });
}

// 每个会话独立进程调用太慢：合并为一次进程，用 /reset 分隔
const input = [];
const order = [];
for (const s of sessions) { input.push('/reset'); for (const [p] of s.turns) input.push(p); order.push(s); }
input.push('/quit');

const r = spawnSync('D:/TaoVm/build/h2r_cpu.exe', [DSB], {
  cwd: 'D:/TaoVm', input: input.join('\n') + '\n', encoding: 'utf8',
  timeout: 1800000, maxBuffer: 1 << 27,
  env: { ...process.env, TAO_TOKENIZER: TOK, TAO_CPU_THREADS: '8' }
});
const out = String(r.stdout || '');
const bodies = [...out.matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map(m => m[1]);

// 每个会话的回复数 = turns 数；按序切分
const stats = {};
let p = 0;
for (const s of order) {
  const replies = bodies.slice(p, p + s.turns.length); p += s.turns.length;
  const got = (replies[replies.length - 1] || '').replace(/\s+/g, '');
  const want = s.turns[s.turns.length - 1][1].replace(/\s+/g, '');
  const t = stats[s.task] || (stats[s.task] = { n: 0, exact: 0, first: 0, bad: 0 });
  t.n++;
  if (got === want) t.exact++;
  else if (want && got.startsWith(want.slice(0, 2))) t.first++;
  if (got.includes('\uFFFD')) t.bad++;
  if (process.env.TAO_SHOW && s.task === (process.env.TAO_TASK || s.task)) {
    console.log('  [' + s.task + '] 期望=' + JSON.stringify(want) + ' 模型=' + JSON.stringify(got.slice(0, 60)));
  }
}
console.log('模型: ' + DSB);
console.log('任务            样本   完全正确   首2字正确   含乱码');
for (const [k, v] of Object.entries(stats)) {
  console.log('  ' + k.padEnd(12) + String(v.n).padStart(5) + String(v.exact).padStart(9) +
              String(v.first).padStart(12) + String(v.bad).padStart(9) +
              '   (' + (100 * v.exact / v.n).toFixed(0) + '% 精确)');
}
