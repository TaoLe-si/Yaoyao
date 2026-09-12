// 对比两条遗忘曲线：控制组 vs T1（排练课程）
// 用法: node scripts/compare_curves.mjs [curveA.tsv] [curveB.tsv]
import fs from 'node:fs';
const A = process.argv[2] || 'D:/TaoVm/build/forget_curve.tsv';
const B = process.argv[3] || 'D:/TaoVm/build/forget_curve_t1.tsv';
const load = (p) => {
  const rows = new Map();
  if (!fs.existsSync(p)) return rows;
  const lines = fs.readFileSync(p, 'utf8').split(/\r?\n/).filter(x => x.trim());
  for (const ln of lines.slice(1)) {
    const c = ln.split(/\t/);
    if (c.length < 4) continue;
    const s = +c[0]; if (!Number.isFinite(s)) continue;
    const v = c.slice(1, 4).map(Number);
    if (v.some(x => !Number.isFinite(x))) continue;
    rows.set(s, { wiki: v[0], reason: v[1], alpaca: v[2] });
  }
  return rows;
};
const a = load(A), b = load(B);
const common = [...a.keys()].filter(s => b.has(s)).sort((x, y) => x - y);
console.log('曲线A 控制组: ' + a.size + ' 点   (' + A + ')');
console.log('曲线B 排练组: ' + b.size + ' 点   (' + B + ')');
console.log('可比点: ' + common.length + '\n');
if (!common.length) { console.log('暂无重叠点。'); process.exit(0); }
const mean = (arr) => arr.reduce((x, y) => x + y, 0) / arr.length;
// 分段：复现控制组阶段边界（步号对齐）：38659 维基止、42659 推理止、50659 终
const seg = (s) => s <= 38659 ? 'S1 维基(11槽)' : (s <= 42659 ? 'S2 推理(4槽)' : 'S3 alpaca(8槽)');
const groups = {};
for (const s of common) { (groups[seg(s)] ??= []).push(s); }
console.log('阶段              点数   wiki A→B              推理 A→B              alpaca A→B');
for (const k of Object.keys(groups)) {
  const ss = groups[k];
  const f = (arr, key) => [mean(ss.map(s => a.get(s)[key])), mean(ss.map(s => b.get(s)[key]))];
  const [wa, wb] = f(ss, 'wiki'), [ra, rb] = f(ss, 'reason'), [aa, ab] = f(ss, 'alpaca');
  const fmt = (x, y) => x.toFixed(4) + '→' + y.toFixed(4) + ' (' + (y > x ? '+' : '') + ((y - x) / x * 100).toFixed(1) + '%)';
  console.log(k.padEnd(16) + String(ss.length).padStart(4) + '  ' + fmt(wa, wb).padEnd(22) + ' ' + fmt(ra, rb).padEnd(22) + ' ' + fmt(aa, ab));
}
const first = common[0], last = common[common.length - 1];
console.log('\n=== 关键裁决：维基相对起点的回退（越小越好）===');
const wk = (m, s) => m.get(s)?.wiki;
const base = wk(a, first);
console.log('  起点 step ' + first + ':  控制 ' + base.toFixed(4));
for (const [name, m] of [['控制组', a], ['排练组', b]]) {
  const w0 = wk(m, first), w1 = wk(m, last);
  console.log('  ' + name + ' 终点 step ' + last + ': wiki ' + w0.toFixed(4) + ' → ' + w1.toFixed(4) +
    '   回退 ' + ((w1 - w0) / w0 * 100).toFixed(1) + '%');
}
console.log('\n=== 学习能力保留（推理+alpaca，越负越好）===');
for (const [name, m] of [['控制组', a], ['排练组', b]]) {
  const r0 = m.get(first).reason, r1 = m.get(last).reason;
  const a0 = m.get(first).alpaca, a1 = m.get(last).alpaca;
  console.log('  ' + name + '  推理 ' + ((r1 - r0) / r0 * 100).toFixed(1) + '%   alpaca ' + ((a1 - a0) / a0 * 100).toFixed(1) + '%');
}
console.log('\n=== 全程曲线（步 / 控制wiki / 排练wiki / 差）===');
for (const s of common) {
  const A_ = a.get(s), B_ = b.get(s);
  console.log('  ' + String(s).padStart(6) + '  ' + A_.wiki.toFixed(4) + '  ' + B_.wiki.toFixed(4) + '  ' + (B_.wiki - A_.wiki >= 0 ? '+' : '') + (B_.wiki - A_.wiki).toFixed(4));
}
