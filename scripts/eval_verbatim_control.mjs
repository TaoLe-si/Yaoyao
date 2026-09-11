// 逐字对照实验：区分「复制/检索」与「计算」，并排除轮次数与提问框架的干扰。
// 三类样本全部取自语料原文，问题与训练时完全一致。
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const EXE = process.env.TAO_CPU_EXE || 'D:/TaoVm/build/h2r_cpu.exe';
const CORPUS = process.env.TAO_CORPUS || 'D:/TaoVm/data/noffn_l6/conversations.txt';
const DSB = process.argv[2];

const lines = fs.readFileSync(CORPUS, 'utf8').split('\n');
const docs = []; let cur = [];
for (const l of lines) {
  if (l === 'DOC') { if (cur.length) docs.push(cur); cur = []; continue; }
  if (l.trim() === '') continue;
  cur.push(l);
}
if (cur.length) docs.push(cur);

function replies(input) {
  const r = spawnSync(EXE, [DSB], {cwd: 'D:/TaoVm', input, encoding: 'utf8', timeout: 900000, maxBuffer: 1 << 26});
  const out = []; let c = null;
  for (const ln of String(r.stdout || '').split(/\r?\n/)) {
    if (ln.startsWith('BEGIN_REPLY ')) { c = []; continue; }
    if (ln.startsWith('END_REPLY ')) { if (c) out.push(c.join('')); c = null; continue; }
    if (c && ln !== 'RESET') c.push(ln);
  }
  return out;
}

const mem = docs.filter(d => d.length === 4 && /^U /.test(d[0]) && /^A /.test(d[1]) && /^U /.test(d[2]) && /^A /.test(d[3]));
const arith = docs.filter(d => d.length === 2 && /^U /.test(d[0]) && /^A /.test(d[1]) && /等于多少/.test(d[0]));
const copy1 = docs.filter(d => d.length === 2 && /^U /.test(d[0]) && /^A /.test(d[1]) && /重复|照抄|原样|按顺序输出/.test(d[0]));
console.log(`样本池: 记忆(2轮)=${mem.length}  算术(1轮)=${arith.length}  复制(1轮)=${copy1.length}`);

const N = 8;
const pick = (a, n) => { const o = []; for (let i = 0; i < n && i < a.length; i++) o.push(a[Math.floor(i * a.length / n)]); return o; };
const M = pick(mem, N), A = pick(arith, N), C = pick(copy1, N);

let inp = []; for (const d of M) inp.push('/reset', d[0].slice(2), d[2].slice(2)); inp.push('/quit');
const rm = replies(inp.join('\n') + '\n');
let mok = 0;
console.log('\n===== 逐字·记忆任务（2轮，答案在上下文里，可复制）=====');
M.forEach((d, i) => {
  const got = (rm[2 * i + 1] || '').trim(), exp = d[3].slice(2).trim(), ok = got === exp;
  if (ok) mok++;
  console.log(`  ${d[0].slice(2, 24).padEnd(24)} 期望=${exp.padEnd(18)} 模型=${JSON.stringify(got).slice(0, 30).padEnd(32)} ${ok ? '✓' : '✗'}`);
});

inp = []; for (const d of A) inp.push('/reset', d[0].slice(2)); inp.push('/quit');
const ra = replies(inp.join('\n') + '\n');
let aok = 0;
console.log('\n===== 逐字·算术任务（1轮，答案需计算）=====');
A.forEach((d, i) => {
  const got = (ra[i] || '').trim(), exp = d[1].slice(2).trim(), ok = got === exp;
  if (ok) aok++;
  console.log(`  ${d[0].slice(2, 24).padEnd(24)} 期望=${exp.padEnd(18)} 模型=${JSON.stringify(got).slice(0, 30).padEnd(32)} ${ok ? '✓' : '✗'}`);
});

let cok = 0;
if (C.length) {
  inp = []; for (const d of C) inp.push('/reset', d[0].slice(2)); inp.push('/quit');
  const rc = replies(inp.join('\n') + '\n');
  console.log('\n===== 逐字·复制任务（1轮，答案在上下文里，可复制）=====');
  C.forEach((d, i) => {
    const got = (rc[i] || '').trim(), exp = d[1].slice(2).trim(), ok = got === exp;
    if (ok) cok++;
    console.log(`  ${d[0].slice(2, 24).padEnd(24)} 期望=${exp.slice(0, 18).padEnd(18)} 模型=${JSON.stringify(got).slice(0, 30).padEnd(32)} ${ok ? '✓' : '✗'}`);
  });
}

console.log(`\n===== 汇总（全部为逐字训练样本）=====`);
console.log(`  记忆(2轮,可复制)  ${mok}/${M.length}`);
console.log(`  算术(1轮,需计算)  ${aok}/${A.length}`);
console.log(`  复制(1轮,可复制)  ${cok}/${C.length}`);
