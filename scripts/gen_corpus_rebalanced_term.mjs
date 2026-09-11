// 重配比语料 + 有序复制答案**带终止符**。
//
// 动因（doc 27 §二 修正）：
//   重配比模型（rebal step_900）的有序复制失败模式细分显示
//     顺序前缀全对 24/40（L=5 时达 4/5），但「顺序对但超量」12/40。
//   ⇒ L≤6 时顺序取得到，主因是**没有终止线索**：答案末尾无 `。`，
//     模型只能靠"数够几项"停下。
//
// 本语料与 gen_corpus_rebalanced.mjs 唯一的差别：
//   有序复制文档的答案由 `A <list>` 改为 `A <list>。`
// 其余（L6 绑定、算术）逐字不变 ⇒ 干净的受控 A/B。
//
// 用法: node scripts/gen_corpus_rebalanced_term.mjs <输出目录> [--all]
//   --all: 给**所有**文档的答案末尾加句号（对照实验，检查是否误伤绑定）
import fs from 'node:fs';

const OUT = process.argv[2] || 'D:/TaoVm/data/rebalanced_term';
const ALL = process.argv.includes('--all');

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

// 只给「有序复制」答案加终止符；--all 时给所有答案加
const isCopy = (u) => u.startsWith('U 请按顺序输出');
const addTerm = (docs) => docs.map((d) => {
  const out = [...d];
  for (let i = 0; i < out.length; i++) {
    if (!out[i].startsWith('A ')) continue;
    const u = out[i - 1] || '';
    if (ALL || isCopy(u)) {
      if (!/[。！？.]$/.test(out[i])) out[i] = out[i] + '。';
    }
  }
  return out;
});

const copyT = addTerm(copy);
const l6T = ALL ? addTerm(l6) : l6;
const arithT = ALL ? addTerm(arith) : arith;

const all = [...l6T, ...arithT, ...copyT];
let seed = 20260921;  // 与 gen_corpus_rebalanced.mjs **相同**，保证 doc 顺序一致
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }

fs.mkdirSync(OUT, { recursive: true });
const lines = [];
for (const d of all) lines.push('DOC', ...d);
fs.writeFileSync(OUT + '/conversations.txt', lines.join('\n') + '\n');

const total = all.length;
console.log(`总计 ${total} 篇  (ALL=${ALL ? '是，所有答案加句号' : '否，仅有序复制'})`);
console.log(`  L6 ${(100 * l6T.length / total).toFixed(2)}%  算术 ${(100 * arithT.length / total).toFixed(2)}%  有序复制 ${(100 * copyT.length / total).toFixed(2)}%`);
console.log('样例（有序复制）: ' + copyT[0].join(' | '));
