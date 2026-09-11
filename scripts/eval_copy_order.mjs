// 有序复制任务评测：三分（完全 / 集合 / 项数），长度 1..8。
// 用法: node scripts/eval_copy_order.mjs <dsb> [每长度样本数]
import {spawnSync} from 'node:child_process';

const EXE = process.env.TAO_DIAG_EXE || 'D:/TaoVm/build/decode_diag.exe';
const DSB = process.argv[2];
const PER = parseInt(process.argv[3] || '5', 10);

const POOL = ['铅笔','草莓','桃子','桌子','绿色','钟表','鱼','北京','广州','兔子','紫色','马',
  '水杯','杭州','白色','重庆','葡萄','橙子','红色','成都','椅子','黄色','地图','笔记本',
  '橡皮','雨伞','猫','狗','蓝色','粉色','香蕉','苹果','西瓜','天津','武汉','书包','西安',
  '鸡','尺子','橙色','上海','石头'];

let seed = 777001;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

const tests = [];
for (let L = 1; L <= 8; L++) {
  for (let k = 0; k < PER; k++) {
    const items = [], used = new Set();
    while (items.length < L) {
      const x = POOL[Math.floor(rnd() * POOL.length)];
      if (used.has(x)) continue;
      used.add(x); items.push(x);
    }
    tests.push([L, items]);
  }
}

const inp = [];
for (const [, items] of tests) inp.push('/reset', '请按顺序输出：' + items.join('、') + '。不要添加其他文字。');
inp.push('/quit');

const r = spawnSync(EXE, [DSB], {cwd: 'D:/TaoVm', input: inp.join('\n') + '\n', encoding: 'utf8', timeout: 900000, maxBuffer: 1 << 26});
const reps = []; let cur = null;
for (const ln of String(r.stdout || '').split(/\r?\n/)) {
  if (ln.startsWith('BEGIN_REPLY ')) { cur = []; continue; }
  if (ln.startsWith('END_REPLY ')) { if (cur) reps.push(cur.join('')); cur = null; continue; }
  if (cur && ln !== 'RESET') cur.push(ln);
}

const byLen = {};
let eT = 0, sT = 0, cT = 0;
const examples = [];
tests.forEach(([L, exp], i) => {
  const raw = (reps[i] || '').trim();
  const got = raw.split('、').map(x => x.replace(/[。\s]/g, '')).filter(Boolean);
  const exact = got.join('、') === exp.join('、');
  const setOk = exp.every(x => got.includes(x));
  const cntOk = got.length === exp.length;
  if (exact) eT++; if (setOk) sT++; if (cntOk) cT++;
  byLen[L] = byLen[L] || {n: 0, e: 0, s: 0, c: 0};
  const b = byLen[L]; b.n++; if (exact) b.e++; if (setOk) b.s++; if (cntOk) b.c++;
  if (examples.length < 10 && L >= 3) examples.push([L, exp.join('、'), raw.slice(0, 40), exact]);
});

console.log('样本 ' + tests.length + ' 条（长度 1..8，每长度 ' + PER + ' 条）');
console.log('\n长度  样本  完全  集合  项数');
for (const L of Object.keys(byLen).sort((a, b) => a - b)) {
  const b = byLen[L];
  console.log(`  ${L}    ${String(b.n).padStart(3)}  ${String(b.e).padStart(4)}  ${String(b.s).padStart(4)}  ${String(b.c).padStart(4)}`);
}
console.log(`\n合计: 完全 ${eT}/${tests.length}  集合 ${sT}/${tests.length}  项数 ${cT}/${tests.length}`);
console.log('\n样例:');
for (const [L, exp, got, ok] of examples)
  console.log(`  L=${L} 期望=${JSON.stringify(exp).padEnd(30)} 模型=${JSON.stringify(got).padEnd(32)} ${ok ? '✓' : '✗'}`);
