// probe3 held-out 评测。三档，全部经逐行泄漏审查（对 conversations.txt 的用户话轮归一化比对）：
//   核心集 core (行 1–20，全部未见)：绑定 remember/recall 12 + 列表复制 4 + 原样复制 4
//   泛化集 gen  (行 21–34 中已确认未见的)：应用题 4 + 算式 5 + 首都换说法 2 + 小时 1
//   生产集 prod (行 23/26/36/41–49，已确认在语料中)：属"生产能力"而非泛化，单列不计入结论
// 用法: TAO_CPU_EXE=<cpu.exe> node scripts/eval_probe3.mjs <label>=<dsb> [...]
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const PROMPTS = 'D:/TaoVm/data/noffn_probe3/heldout_prompts.txt';
const EXE = process.env.TAO_CPU_EXE || 'D:/TaoVm/build/yaoyao_cpu_v01.exe';
const LEAKED = new Set([23, 26, 36, 41, 42, 43, 44, 45, 46, 47, 48, 49]); // 已确认出现在训练语料

const lines = fs.readFileSync(PROMPTS, 'utf8').split(/\r?\n/)
  .map(s => s.trim()).filter(s => s && s !== '/reset' && s !== '/quit');

const num = s => {
  const m = (s || '').replace(/[０-９]/g, c => String.fromCharCode(c.charCodeAt(0) - 0xFEE0)).match(/-?\d+/);
  return m ? +m[0] : NaN;
};

const items = [];
const push = (i, cat, kind, want, ok) => items.push({i, cat, kind, want, ok});

for (let i = 0; i < lines.length; i++) {
  const L = lines[i], n0 = i + 1;
  let m;
  if ((m = L.match(/名字叫([^，,]+)[，,]\s*我喜欢([^。]+)/))) {
    const name = m[1], like = m[2];
    push(n0, 'binding', 'remember', `${name}+${like}`, r => r.includes(name) && r.includes(like));
    const R = lines[i + 1] || '';
    if (/我叫什么名字/.test(R))
      push(n0 + 1, 'binding', 'recall', `${name}+${like}`, r => r.includes(name) && r.includes(like));
  } else if ((m = L.match(/请按顺序输出：(.+?)。/))) {
    const want = m[1];
    push(n0, 'copy', 'list', want,
      r => want.split('、').every((x, k, a) => r.indexOf(x) >= 0 && (k === 0 || r.indexOf(x) > r.indexOf(a[k - 1]))));
  } else if ((m = L.match(/请重复这句话：(.+?)。/))) {
    const want = m[1];
    push(n0, 'copy', 'repeat', want, r => r.includes(want));
  } else if ((m = L.match(/小明有(\d+)个苹果，又买了(\d+)个/))) {
    const want = String(+m[1] + +m[2]);
    push(n0, 'reason', 'word', want, r => num(r) === +want);
  } else if ((m = L.match(/(\d+)\s*([+\-×])\s*(\d+)/))) {
    const a = +m[1], b = +m[3];
    const want = String(m[2] === '+' ? a + b : m[2] === '-' ? a - b : a * b);
    push(n0, 'reason', 'expr', want, r => num(r) === +want);
  } else if (/德国的首都是/.test(L)) {
    push(n0, 'fact', 'cap', '柏林', r => r.includes('柏林'));
  } else if (/韩国的首都是/.test(L)) {
    push(n0, 'fact', 'cap', '首尔', r => r.includes('首尔'));
  } else if (/一天有多少小时/.test(L)) {
    push(n0, 'fact', 'hours', '24', r => num(r) === 24);
  }
}

const tier = it => (it.i <= 20 ? 'core' : (LEAKED.has(it.i) ? 'prod' : 'gen'));
const CORE = items.filter(it => tier(it) === 'core');
const GEN = items.filter(it => tier(it) === 'gen');
const PROD = items.filter(it => tier(it) === 'prod');

function run(dsb) {
  const input = fs.readFileSync(PROMPTS, 'utf8');
  const r = spawnSync(EXE, [dsb], {cwd: 'D:/TaoVm', input, encoding: 'utf8', timeout: 900000, maxBuffer: 1 << 26});
  const out = String(r.stdout || '') + String(r.stderr || '');
  const byN = {}; let cur = null;
  for (const ln of out.split(/\r?\n/)) {
    if (ln.startsWith('BEGIN_REPLY ')) { cur = {n: +ln.slice(12).trim(), t: []}; continue; }
    if (ln.startsWith('END_REPLY ')) { if (cur) { byN[cur.n] = cur.t.join(''); cur = null; } continue; }
    if (cur && ln !== 'RESET') cur.t.push(ln);
  }
  return byN;
}

const tally = (list, byN, key) => {
  const c = {};
  for (const it of list) {
    const r = byN[it.i];
    const k = key(it);
    (c[k] ||= {p: 0, t: 0}); c[k].t++; if (r !== undefined && it.ok(r)) c[k].p++;
  }
  return c;
};
const fmt = (c, k) => c[k] ? `${c[k].p}/${c[k].t}` : '-';

const rows = [];
for (const arg of process.argv.slice(2)) {
  const eq = arg.indexOf('=');
  const label = arg.slice(0, eq), dsb = arg.slice(eq + 1);
  const byN = run(dsb);

  console.log(`\n########## ${label} ##########`);
  for (const it of CORE) {
    const r = byN[it.i] !== undefined ? byN[it.i] : '(无回复)';
    console.log(`  ${it.ok(r) ? 'PASS' : 'FAIL'} [${it.kind}] 期望=${it.want}  得到=${JSON.stringify(r).slice(0, 46)}`);
  }
  for (const it of GEN) {
    const r = byN[it.i] !== undefined ? byN[it.i] : '(无回复)';
    console.log(`  ${it.ok(r) ? 'PASS' : 'FAIL'} [${it.kind}·泛化] 期望=${it.want}  得到=${JSON.stringify(r).slice(0, 46)}`);
  }

  const kc = tally(CORE, byN, it => it.kind);
  const gc = tally(GEN, byN, it => it.kind);
  const pc = tally(PROD, byN, it => it.kind);
  const sum = c => Object.values(c).reduce((a, b) => a + b.p, 0) + '/' + Object.values(c).reduce((a, b) => a + b.t, 0);

  console.log(`  核心集 core  绑定 ${fmt(tally(CORE, byN, it => it.cat), 'binding')}  复制 ${fmt(tally(CORE, byN, it => it.cat), 'copy')}  合计 ${sum(kc)}`);
  console.log(`    细分  remember ${fmt(kc, 'remember')}  recall ${fmt(kc, 'recall')}  list ${fmt(kc, 'list')}  repeat ${fmt(kc, 'repeat')}`);
  console.log(`  泛化集 gen   ${sum(gc)}   细分  word ${fmt(gc, 'word')}  expr ${fmt(gc, 'expr')}  cap ${fmt(gc, 'cap')}  hours ${fmt(gc, 'hours')}`);
  console.log(`  生产集 prod  ${sum(pc)} （已确认在语料中，不计入结论）`);
  rows.push({label, kc, gc, pc, core: sum(kc), gen: sum(gc)});
}

console.log('\n\n===== 汇总 =====');
for (const r of rows) {
  console.log(`${r.label}\t核心:${r.core}\t泛化:${r.gen}`);
  console.log(`\tremember:${fmt(r.kc, 'remember')}\trecall:${fmt(r.kc, 'recall')}\tlist:${fmt(r.kc, 'list')}\trepeat:${fmt(r.kc, 'repeat')}`);
  console.log(`\tword:${fmt(r.gc, 'word')}\texpr:${fmt(r.gc, 'expr')}\tcap:${fmt(r.gc, 'cap')}\thours:${fmt(r.gc, 'hours')}`);
}
