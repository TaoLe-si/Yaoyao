// 真实语料 held-out 评测。
// 指标（不依赖任何模板假设）:
//   empty      空回复率
//   garbled    含 U+FFFD 乱码率
//   loop       退化复读率（同一 8+ 字片段反复出现 3 次以上）
//   bigramF1   与参考答案的字符 bigram 重合度（中文松散相关性）
//   lenRatio   生成长度 / 参考长度
// 用法: node scripts/eval_real.mjs <model.dsb> <heldout.jsonl> [n]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';

const DSB = process.argv[2];
const HL = process.argv[3] || 'D:/TaoVm/data/alpaca_heldout.jsonl';
const N = Number(process.argv[4] || 200);
const recs = fs.readFileSync(HL, 'utf8').split('\n').filter(Boolean).slice(0, N)
  .map(l => JSON.parse(l));

const input = [];
for (const r of recs) { input.push('/reset'); input.push(r.user); }
input.push('/quit');
const p = spawnSync('D:/TaoVm/build/h2r_cpu.exe', [DSB], {
  cwd: 'D:/TaoVm', input: input.join('\n') + '\n', encoding: 'utf8',
  timeout: 1800000, maxBuffer: 1 << 27,
  env: { ...process.env, TAO_TOKENIZER: process.env.TAO_TOKENIZER || 'D:/TaoVm/build/tok_real_v1.bbp', TAO_CPU_THREADS: '8' }
});
const bodies = [...String(p.stdout || '').matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map(m => m[1]);

const bigrams = s => { const o = []; const t = s.replace(/\s+/g, '');
  for (let i = 0; i + 1 < t.length; i++) o.push(t.slice(i, i + 2)); return o; };
const isLoop = s => { const t = s.replace(/\s+/g, '');
  if (t.length < 24) return false;
  for (let L = 8; L <= 16; L++) for (let i = 0; i + L <= t.length; i++) {
    const seg = t.slice(i, i + L); let c = 0, j = i;
    while ((j = t.indexOf(seg, j)) !== -1) { c++; j += L; if (c >= 3) return true; } }
  return false; };

let empty = 0, garbled = 0, loop = 0, f1sum = 0, lrsum = 0, n = 0;
const samples = [];
for (const r of recs) {
  const got = (bodies[n] || '').replace(/\s+/g, '');
  const ref = (r.ref || '').replace(/\s+/g, '');
  if (!got) empty++;
  if (got.includes('\uFFFD')) garbled++;
  if (isLoop(got)) loop++;
  const g = bigrams(got), f = bigrams(ref);
  const G = new Set(g), F = new Set(f);
  let inter = 0; for (const x of G) if (F.has(x)) inter++;
  const f1 = (G.size && F.size) ? (2 * inter) / (G.size + F.size) : 0;
  f1sum += f1; lrsum += ref.length ? got.length / ref.length : 0;
  const SHOW = Number(process.env.TAO_SHOW || 0);
  if (SHOW && n < SHOW) samples.push('【Q】' + r.user + '\n【A】' + got);
  n++;
}
console.log('模型: ' + DSB + '   rep_pen=' + (process.env.TAO_REP_PEN || '1.0(默认)'));
console.log('held-out n=' + n);
console.log('  空回复   = ' + empty + '  (' + (100 * empty / n).toFixed(1) + '%)');
console.log('  乱码     = ' + garbled + '  (' + (100 * garbled / n).toFixed(1) + '%)');
console.log('  退化复读 = ' + loop + '  (' + (100 * loop / n).toFixed(1) + '%)');
console.log('  bigramF1 = ' + (f1sum / n).toFixed(4));
console.log('  长度比   = ' + (lrsum / n).toFixed(3));
for (const s of samples) console.log(s + '\n');
