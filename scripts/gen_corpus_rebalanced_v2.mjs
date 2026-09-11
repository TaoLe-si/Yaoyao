// 修复版语料：在 rebalanced_term 基础上，把「原样复制」从 15 个唯一句 ×40
// 换成 2615 个唯一句（doc 28 §二 的修复）。
//
// 其余全部保持不变（绑定 61%、算术 20%、有序复制 19% 的配比与内容逐字相同），
// 便于与 build/rebal_term 做受控对照。
//
// 用法: node scripts/gen_corpus_rebalanced_v2.mjs <输出目录>
import fs from 'node:fs';

const OUT = process.argv[2] || 'D:/TaoVm/data/rebalanced_v2';
const SENT = fs.readFileSync('D:/TaoVm/data/verbatim_diverse/sentences.txt', 'utf8')
  .split('\n').map(s => s.trim()).filter(Boolean);

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

const base = readDocs('D:/TaoVm/data/rebalanced_term/conversations.txt');
const kept = base.filter(d => !d.some(l => l.startsWith('U 请重复这句话')));
const removed = base.length - kept.length;

// 每个唯一句只用一次（1.0× 重复），最大化多样性
const verbatim = SENT.map(s => [`U 请重复这句话：${s}`, `A ${s}`]);

const all = [...kept, ...verbatim];
let seed = 20260923;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }

fs.mkdirSync(OUT, { recursive: true });
const lines = [];
for (const d of all) lines.push('DOC', ...d);
fs.writeFileSync(OUT + '/conversations.txt', lines.join('\n') + '\n');

const total = all.length;
console.log(`移除旧原样复制 ${removed} 篇，加入新唯一句 ${verbatim.length} 篇`);
console.log(`总计 ${total} 篇`);
console.log(`  原样复制 ${verbatim.length} 篇 = ${(100 * verbatim.length / total).toFixed(2)}%（唯一率 ${(100 * verbatim.length / verbatim.length).toFixed(0)}%）`);
