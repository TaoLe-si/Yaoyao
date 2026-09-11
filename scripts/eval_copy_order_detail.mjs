// 有序复制「失败模式细分」评测。
//
// 动机：总表只报「完全」，无法区分两类完全不同的失败：
//   (A) 内容/顺序错  —— 集合也不对
//   (B) 顺序全对但超量（多吐一个/不停）—— 集合对、项数错
// 这两类的修复手段完全不同：
//   (A) 需要顺序通道（架构）
//   (B) 只需要「终止线索」（数据：答案末尾加句号即可）
//
// 用法: node scripts/eval_copy_order_detail.mjs <dsb> [每长度样本数]
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
for (let L = 1; L <= 8; L++) for (let k = 0; k < PER; k++) {
  const items = [], used = new Set();
  while (items.length < L) { const x = POOL[Math.floor(rnd() * POOL.length)]; if (used.has(x)) continue; used.add(x); items.push(x); }
  tests.push([L, items]);
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

// 前缀正确长度：模型输出与期望从头开始连续相同的最长项数
function prefixLen(got, exp) { let i = 0; while (i < exp.length && i < got.length && got[i] === exp[i]) i++; return i; }

const byLen = {};
let eT = 0, sT = 0, cT = 0, pT = 0, overT = 0;
const samples = [];
tests.forEach(([L, exp], i) => {
  const raw = (reps[i] || '').trim();
  const got = raw.split('、').map(x => x.replace(/[。\s]/g, '')).filter(Boolean);
  const exact = got.join('、') === exp.join('、');
  const setOk = exp.every(x => got.includes(x));
  const cntOk = got.length === exp.length;
  const pf = prefixLen(got, exp);
  const over = got.length > exp.length && setOk;
  if (exact) eT++; if (setOk) sT++; if (cntOk) cT++;
  if (pf === L) pT++; if (over) overT++;
  const b = byLen[L] = byLen[L] || {n: 0, e: 0, s: 0, c: 0, p: 0, o: 0};
  b.n++; if (exact) b.e++; if (setOk) b.s++; if (cntOk) b.c++; if (pf === L) b.p++; if (over) b.o++;
  if (samples.length < 16 && L >= 3 && !exact) samples.push([L, exp.join('、'), raw.slice(0, 60), `前缀${pf}/${L} 项数${got.length} 集合${setOk ? '✓' : '✗'}`]);
});

console.log('模型: ' + DSB);
console.log('样本 ' + tests.length + ' 条（长度 1..8，每长度 ' + PER + '）\n');
console.log('长度  样本  完全  集合  项数  顺序前缀  顺序对但超量');
for (const L of Object.keys(byLen).sort((a, b) => a - b)) {
  const b = byLen[L];
  console.log(`  ${L}    ${String(b.n).padStart(3)}  ${String(b.e).padStart(4)}  ${String(b.s).padStart(4)}  ${String(b.c).padStart(4)}  ${String(b.p).padStart(8)}  ${String(b.o).padStart(12)}`);
}
console.log(`\n合计: 完全 ${eT}/${tests.length}  集合 ${sT}  项数 ${cT}  顺序前缀全对 ${pT}  顺序对但超量 ${overT}`);
console.log('\n失败样例（L≥3）:');
for (const [L, exp, got, note] of samples) {
  console.log(`  L=${L} ${note}`);
  console.log(`      期望=${JSON.stringify(exp)}`);
  console.log(`      模型=${JSON.stringify(got)}`);
}
