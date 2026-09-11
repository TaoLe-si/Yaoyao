// 受控语料：纯「有序复制」任务，长度 1..8。
// 目的（doc 22）：判定「顺序丢失」是架构容量问题还是数据配比问题。
// 若本语料上顺序正确率仍接近 0，则与数据配比无关，是架构缺陷。
import fs from 'node:fs';

const OUT = process.argv[2] || 'D:/TaoVm/data/copy_order/conversations.txt';
const N = parseInt(process.argv[3] || '6000', 10);
const SEED = 20260913;

let seed = SEED;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const pick = (a) => a[Math.floor(rnd() * a.length)];

// 42 个互不相同的物品名（与 L6 列表项池一致，便于对照）
const POOL = ['铅笔','草莓','桃子','桌子','绿色','钟表','鱼','北京','广州','兔子','紫色','马',
  '水杯','杭州','白色','重庆','葡萄','橙子','红色','成都','椅子','黄色','地图','笔记本',
  '橡皮','雨伞','猫','狗','蓝色','粉色','香蕉','苹果','西瓜','天津','武汉','书包','西安',
  '鸡','尺子','橙色','上海','石头'];

const docs = [];
const lenCount = {};
for (let i = 0; i < N; i++) {
  // 长度 1..8 均匀分布（这是修复「终止/计数」缺陷的关键：让模型见到各种长度）
  const L = 1 + Math.floor(rnd() * 8);
  lenCount[L] = (lenCount[L] || 0) + 1;
  const items = [];
  const used = new Set();
  while (items.length < L) {
    const x = pick(POOL);
    if (used.has(x)) continue;   // 同一条内不重复，保证顺序是唯一解
    used.add(x); items.push(x);
  }
  const list = items.join('、');
  docs.push(['U 请按顺序输出：' + list + '。不要添加其他文字。', 'A ' + list]);
}

const lines = [];
for (const [u, a] of docs) { lines.push('DOC', u, a, ''); }
fs.mkdirSync(OUT.replace(/\/[^/]+$/, ''), { recursive: true });
fs.writeFileSync(OUT, lines.join('\n'), 'utf8');

console.log('生成 ' + docs.length + ' 篇纯有序复制文档 -> ' + OUT);
console.log('长度分布: ' + Object.keys(lenCount).sort((a, b) => a - b).map(k => k + ':' + lenCount[k]).join('  '));
const uniq = new Set(docs.map(d => d[0]));
console.log('唯一 U 行 ' + uniq.size + ' / ' + docs.length);
