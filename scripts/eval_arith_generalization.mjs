// 算术泛化测试：区分「查表记忆」与「学到算法」。
// 语料中 1..20 的 a+b 只覆盖 320/400，剩余 80 个组合从未出现 → 天然泛化测试集。
// 三组：语料内(对照) / 组合泛化(两操作数各自都见过，但组合未见) / 外推(21..40)
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const EXE = process.env.TAO_CPU_EXE || 'D:/TaoVm/build/h2r_cpu.exe';
const CORPUS = process.env.TAO_CORPUS || 'D:/TaoVm/data/noffn_l6/conversations.txt';
const dsbArg = process.argv[2];
const OP = process.argv[3] || '+';

const c = fs.readFileSync(CORPUS, 'utf8');
const esc = OP === '+' ? '\\+' : (OP === '×' ? '×' : OP);
const seen = new Set();
for (const m of c.matchAll(new RegExp(`(\\d+)${esc}(\\d+)等于多少`, 'g'))) seen.add(m[1] + OP + m[2]);
const has = (a, b) => seen.has(a + OP + b);
console.log(`语料中 ${OP} 数对: ${seen.size} 种`);

const ctrl = [], comp = [], extra = [];
for (let a = 1; a <= 20 && ctrl.length < 15; a++) for (let b = 1; b <= 20 && ctrl.length < 15; b++) if (has(a, b)) ctrl.push([a, b]);
for (let a = 1; a <= 20 && comp.length < 15; a++) for (let b = 1; b <= 20 && comp.length < 15; b++) if (!has(a, b)) comp.push([a, b]);
for (let a = 21; a <= 40 && extra.length < 12; a++) for (let b = 21; b <= 40 && extra.length < 12; b++) extra.push([a, b]);

const all = [...ctrl.map(x => [x, 'ctrl']), ...comp.map(x => [x, 'comp']), ...extra.map(x => [x, 'extra'])];
const L = [];
for (const [[a, b]] of all) L.push('/reset', `${a}${OP}${b}等于多少？`);
L.push('/quit');

const r = spawnSync(EXE, [dsbArg], {cwd: 'D:/TaoVm', input: L.join('\n') + '\n', encoding: 'utf8', timeout: 900000, maxBuffer: 1 << 26});
const reps = []; let cur = null;
for (const ln of String(r.stdout || '').split(/\r?\n/)) {
  if (ln.startsWith('BEGIN_REPLY ')) { cur = []; continue; }
  if (ln.startsWith('END_REPLY ')) { if (cur) reps.push(cur.join('')); cur = null; continue; }
  if (cur && ln !== 'RESET') cur.push(ln);
}
console.log(`问题数=${all.length} 回复数=${reps.length}`);
const truth = (a, b) => OP === '+' ? a + b : a * b;
const stat = {ctrl: [0, 0], comp: [0, 0], extra: [0, 0]}, detail = {ctrl: [], comp: [], extra: []};
all.forEach(([[a, b], g], i) => {
  const rep = reps[i] || '';
  const m = rep.match(/-?\d+/);                 // 答案就是纯数字
  const got = m ? parseInt(m[0], 10) : null;
  const t = truth(a, b);
  const ok = got === t;
  stat[g][1]++; if (ok) stat[g][0]++;
  if (detail[g].length < 10) detail[g].push(`${a}${OP}${b}=${t}→${got === null ? '无' : got}${ok ? '✓' : '✗'}`);
});
console.log(`\n===== 算术泛化测试 (${OP}) =====`);
console.log(`语料内(对照)    ${stat.ctrl[0]}/${stat.ctrl[1]}   ${detail.ctrl.join('  ')}`);
console.log(`组合泛化(未见)  ${stat.comp[0]}/${stat.comp[1]}   ${detail.comp.join('  ')}`);
console.log(`外推(21..40)    ${stat.extra[0]}/${stat.extra[1]}   ${detail.extra.join('  ')}`);
console.log('\n原始回复(前8): ' + reps.slice(0, 8).map(x => JSON.stringify(x)).join(' '));
