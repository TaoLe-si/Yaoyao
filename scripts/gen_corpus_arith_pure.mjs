// 受控语料：纯算术（无任何对话/绑定任务），用于区分
//   「数据配比不足」 vs 「架构无法承载数值关系」。
//
// 背景（doc 25 之后的实测）：在 noffn_l6 混合语料上，算术只占 4.03%，
// 模型在**语料内见过的 320 个数对**上也只有 1/15(+)、0/15(×)，
// 输出恒为少数常量（11/20/14/15/2）——典型的边缘分布坍缩。
//
// 本实验：把算术抽出来单独训练。若纯算术语料能学会 → 是配比问题（课程式长训可解）；
// 若仍学不会 → 是架构/三值量化无法承载数值关系。
//
// 设计：a,b ∈ 1..20 的 400 个数对中只用 320 个（留 80 个从未出现 → 组合泛化测试），
// 与 scripts/eval_arith_generalization.mjs 的 seen 集口径一致。
//
// 用法: node scripts/gen_corpus_arith_pure.mjs <输出目录> [重复倍数]
import fs from 'node:fs';

const OUT = process.argv[2] || 'D:/TaoVm/data/arith_pure';
const REP = Number(process.argv[3] || 10);

let seed = 20260920;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

function pickSeen() {
  const all = [];
  for (let a = 1; a <= 20; a++) for (let b = 1; b <= 20; b++) all.push([a, b]);
  shuffle(all);
  return all.slice(0, 320);
}
const seenAdd = pickSeen(), seenMul = pickSeen();

const FORMS = [
  (a, o, b) => [a + o + b + '等于多少？', String(o === '+' ? a + b : a * b)],
  (a, o, b) => ['只输出数字：' + a + o + b + '等于多少？', String(o === '+' ? a + b : a * b)],
  (a, o, b) => ['计算' + a + o + b + '。', String(o === '+' ? a + b : a * b)],
  (a, o, b) => [a + o + b + '是多少？', String(o === '+' ? a + b : a * b)],
];

const docs = [];
for (let r = 0; r < REP; r++) {
  for (const [op, pairs] of [['+', seenAdd], ['×', seenMul]])
    for (const [a, b] of pairs) {
      const f = FORMS[Math.floor(rnd() * FORMS.length)];
      const [q, ans] = f(a, op, b);
      docs.push(['DOC', 'U ' + q, 'A ' + ans]);
    }
}
shuffle(docs);

fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(OUT + '/conversations.txt', docs.flat().join('\n') + '\n');
console.log('docs=' + docs.length + ' seenAdd=' + seenAdd.length + ' seenMul=' + seenMul.length);
