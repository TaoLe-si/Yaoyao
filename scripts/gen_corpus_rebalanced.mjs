// 重配比混合语料：验证「数据配比就是缺陷 2/3 的修复手段」。
//
// 依据 doc 26 的受控实验：
//   · 混合 L6 语料：绑定 77.03% / 算术 4.03% / 有序复制 5.1%
//     → 算术语料内 1/15(+)、0/15(×)；6 项有序复制连训练样本都做不对
//   · 纯算术语料 600 步 → 语料内 14/15(+)、14/15(×)
//   · 纯复制语料 → 顺序前缀 10→17/40
// 三者根因都指向配比。本语料把算术/复制提到与绑定同量级，其余不变。
//
// 配比：L6 19520 篇(61%) + 算术 6400 篇(20%) + 有序复制 6000 篇(19%)
//
// 用法: node scripts/gen_corpus_rebalanced.mjs <输出目录>
import fs from 'node:fs';

const OUT = process.argv[2] || 'D:/TaoVm/data/rebalanced';

// 按 DOC 切分文档；返回 [[行...], ...]
function readDocs(path) {
  const docs = []; let cur = [];
  for (const raw of fs.readFileSync(path, 'utf8').split('\n')) {
    const line = raw.endsWith('\r') ? raw.slice(0, -1) : raw;
    if (line === 'DOC') { if (cur.length) docs.push(cur); cur = []; continue; }
    if (!line || line[0] === '#') continue;
    cur.push(line);
  }
  if (cur.length) docs.push(cur);
  return docs;
}

const l6 = readDocs('D:/TaoVm/data/noffn_l6/conversations.txt');
const arith = readDocs('D:/TaoVm/data/arith_pure/conversations.txt');
const copy = readDocs('D:/TaoVm/data/copy_order/conversations.txt');

console.log(`L6=${l6.length}  算术=${arith.length}  有序复制=${copy.length}`);

// 全部保留：三者相加即为目标配比，不做二次采样
const all = [...l6, ...arith, ...copy];

// 打散，避免任务块状排列；训练器每 epoch 还会再 shuffle
let seed = 20260921;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }

fs.mkdirSync(OUT, { recursive: true });
const out = [];
for (const d of all) { out.push('DOC', ...d); }
fs.writeFileSync(OUT + '/conversations.txt', out.join('\n') + '\n');

const total = all.length;
console.log(`总计 ${total} 篇`);
console.log(`  L6(绑定为主) ${(100 * l6.length / total).toFixed(2)}%`);
console.log(`  算术         ${(100 * arith.length / total).toFixed(2)}%`);
console.log(`  有序复制     ${(100 * copy.length / total).toFixed(2)}%`);
