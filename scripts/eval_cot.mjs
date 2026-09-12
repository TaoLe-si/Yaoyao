// 思考链 held-out：从回复里取「答案：」后的内容，否则取最后一个数。
// 用法: node scripts/eval_cot.mjs [model.dsb] [heldout.jsonl]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';

const DSB = process.argv[2] || 'D:/TaoVm/build/L1_qa_cot/final.dsb';
const HL = process.argv[3] || 'D:/TaoVm/data/qa_cot/heldout_cot.jsonl';
const recs = fs.readFileSync(HL, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const input = [];
for (const r of recs) { input.push('/reset'); input.push(r.user); }
input.push('/quit');
const p = spawnSync('D:/TaoVm/build/h2r_cpu.exe', [DSB, '--max-out', '256', '--rep-pen', '1.0'], {
  cwd: 'D:/TaoVm', input: input.join('\n') + '\n', encoding: 'utf8',
  timeout: 1800000, maxBuffer: 1 << 27,
  env: { ...process.env, TAO_TOKENIZER: process.env.TAO_TOKENIZER || 'D:/TaoVm/build/tok_qa.bbp', TAO_CPU_THREADS: '8' },
});
const bodies = [...String(p.stdout || '').matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map((m) => m[1]);

function extract(s) {
  const t = String(s || '').replace(/\s+/g, '');
  const m = t.match(/答案[：:]([^答案思考]+)$/) || t.match(/答案[：:](.+)/);
  if (m) return m[1].replace(/。+$/, '');
  const nums = t.match(/-?\d+(?:_\d+)?(?:\/\d+)?(?:\.\d+)?/g);
  return nums && nums.length ? nums[nums.length - 1] : t.slice(-24);
}
function norm(s) { return String(s || '').replace(/\s+/g, '').replace(/_/g, '').replace(/。+$/, ''); }

const by = {};
let n = 0, think = 0, empty = 0, loop = 0;
const isLoop = (s) => {
  const t = s.replace(/\s+/g, '');
  if (t.length < 24) return false;
  for (let L = 8; L <= 16; L++) for (let i = 0; i + L <= t.length; i++) {
    const seg = t.slice(i, i + L); let c = 0, j = i;
    while ((j = t.indexOf(seg, j)) !== -1) { c++; j += L; if (c >= 3) return true; }
  }
  return false;
};
const samples = [];
for (const r of recs) {
  const got = bodies[n] || '';
  const kind = r.kind || 'cot';
  (by[kind] ||= { p: 0, t: 0 });
  by[kind].t++;
  if (!got.trim()) empty++;
  if (got.includes('思考') || got.includes('计算')) think++;
  if (isLoop(got)) loop++;
  const ok = norm(extract(got)) === norm(r.ref) || got.includes(String(r.ref));
  if (ok) by[kind].p++;
  if (samples.length < 8) samples.push('【Q】' + r.user + '\n【A】' + got.replace(/\s+/g, ' ').slice(0, 180) + '\n【want】' + r.ref + (ok ? ' OK' : ' BAD'));
  n++;
}
console.log('模型: ' + DSB);
console.log('cot held-out n=' + n + ' 空=' + empty + ' 复读=' + loop + ' 含思考链=' + think + '/' + n);
for (const k of Object.keys(by)) {
  const x = by[k];
  console.log('  ' + k + '  ' + x.p + '/' + x.t + '  (' + (100 * x.p / x.t).toFixed(1) + '%)');
}
for (const s of samples) console.log('\n' + s);
