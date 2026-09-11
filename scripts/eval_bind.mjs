// binding 专项评测：多轮会话内绑定一个变量，再召回。
// 名字全部来自 held-out（与训练名零交集），所以考的是绑定而非背诵。
// 用法: node scripts/eval_bind.mjs <model.dsb> [n]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';

const DSB = process.argv[2];
const N = Number(process.argv[3] || 120);
const TOK = 'D:/TaoVm/build/tok_digit_v1.bbp';
const recs = fs.readFileSync('D:/TaoVm/data/bind_heldout.txt', 'utf8')
  .split('\n').filter(Boolean).slice(0, N).map(l => JSON.parse(l));

const input = [];
for (const r of recs) { input.push('/reset'); for (const t of r.turns) input.push(t.slice(2)); }
input.push('/quit');

const p = spawnSync('D:/TaoVm/build/h2r_cpu.exe', [DSB], {
  cwd: 'D:/TaoVm', input: input.join('\n') + '\n', encoding: 'utf8',
  timeout: 1800000, maxBuffer: 1 << 27,
  env: { ...process.env, TAO_TOKENIZER: TOK, TAO_CPU_THREADS: '8' }
});
const bodies = [...String(p.stdout || '').matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map(m => m[1]);
let exact = 0, contains = 0, p2 = 0;
const fails = [];
for (const r of recs) {
  const replies = bodies.slice(p2, p2 + 4); p2 += 4;          // 4 轮
  const got = (replies[3] || '').replace(/\s+/g, '');
  const want = r.name;
  if (got === want + '。' || got === want || got === '你叫' + want + '。') exact++;
  else if (got.includes(want)) contains++;
  else if (fails.length < 5) fails.push('期望=' + want + ' 实际=' + JSON.stringify(got.slice(0, 24)));
}
console.log('模型: ' + DSB);
console.log('binding held-out (名字与训练零交集) n=' + recs.length);
console.log('  完全正确 = ' + exact + '/' + recs.length + '  (' + (100 * exact / recs.length).toFixed(1) + '%)');
console.log('  含正确名字 = ' + (exact + contains) + '/' + recs.length);
for (const f of fails) console.log('  ✗ ' + f);
