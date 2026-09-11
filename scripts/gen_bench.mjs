// 生成大样本 held-out 能力基准（bench_v1）。
// 原则：
//   1) 元素（名字/喜好/列表项）取自训练语料自身的词表 —— 保证测的是"绑定"不是"分词"；
//   2) 组合（名字×喜好）必须是语料中从未出现的 —— 保证测的是泛化不是记忆；
//   3) 每一条生成的话轮都逐条与语料归一化比对，任何命中即报错退出 —— 防止重演 probe2 泄漏事故。
// 用法: node scripts/gen_bench.mjs [语料] [输出目录]
import fs from 'node:fs';

const CORPUS = process.argv[2] || 'D:/TaoVm/data/noffn_probe3/conversations.txt';
const OUT = process.argv[3] || 'D:/TaoVm/data/bench_v1';

const CN = '\\u4e00-\\u9fa5';
const COLORS = ['红色','橙色','黄色','绿色','青色','蓝色','紫色','黑色','白色','灰色','粉色','棕色','银色','金色'];

// 归一化：去掉角色前缀与全部空白，用于泄漏比对
const norm = s => s.replace(/^[UAS]\s+/, '').replace(/\s/g, '');

const raw = fs.readFileSync(CORPUS, 'utf8');
const lines = raw.split(/\r?\n/).map(l => l.replace(/^"|"$/g, ''));
const corpusSet = new Set(lines.map(norm).filter(Boolean));

// ---------- 1) 抽取词表 ----------
const names = new Set(), likings = new Set(), listItems = new Set(), repeatSent = [];
const seenPair = new Set();   // 语料中出现过的 名字×喜好 组合
for (const l of lines) {
  let m;
  if ((m = l.match(new RegExp('名字叫([' + CN + ']{2,4})[，,]\\s*我喜欢([' + CN + ']{1,3})')))) {
    names.add(m[1]); likings.add(m[2]); seenPair.add(m[1] + '|' + m[2]);
  }
  if ((m = l.match(/请按顺序输出：(.+?)。/))) m[1].split('、').forEach(x => listItems.add(x));
  if ((m = l.match(/请重复这句话：(.+?)。/))) repeatSent.push(m[1]);
}
const NAME = [...names].sort(), LIKE = [...likings].sort();
const COLOR = LIKE.filter(x => COLORS.includes(x));
const NONCOLOR = LIKE.filter(x => !COLORS.includes(x));
const ITEM = [...listItems].sort();

// 确定性伪随机（不用 Math.random，保证基准可复现）
let seed = 20260911;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

// ---------- 2) 生成条目 ----------
const items = [];
// turn = {t: 文本, novel: 是否必须完全未见于语料}
// 注意：问句模板（"我叫什么名字？喜欢什么颜色？"）**必须**出现在语料里，模型才学得会这个行为，
//       所以模板话轮 novel=false；只有事实注入与复制目标必须新颖。
const T = (t, novel = true) => ({ t, novel });
const push = (cat, mode, expect, turns) => items.push({ cat, mode, expect, turns });

// 生成 n 个"语料中从未出现"的名字×喜好组合
function novelPairs(pool, want) {
  const out = [], tried = new Set();
  let guard = 0;
  while (out.length < want && guard++ < want * 200) {
    const n = NAME[(rnd() * NAME.length) | 0], k = pool[(rnd() * pool.length) | 0];
    const key = n + '|' + k;
    if (seenPair.has(key) || tried.has(key)) continue;
    tried.add(key); out.push([n, k]);
  }
  return out;
}

// A. 绑定回忆（颜色）
for (const [n, k] of novelPairs(COLOR, 200))
  push('A_bind_color', 'all', [n, k],
    [T(`我的名字叫${n}，我喜欢${k}。请记住。`), T('我叫什么名字？喜欢什么颜色？', false)]);

// B. 绑定回忆（非颜色喜好）
for (const [n, k] of novelPairs(NONCOLOR, 150))
  push('B_bind_like', 'all', [n, k],
    [T(`我的名字叫${n}，我喜欢${k}。请记住。`), T('我叫什么名字？喜欢什么？', false)]);

// C. 换说法（同一事实，不同问法）
const REPHRASE = ['请问我的名字是什么？我喜欢什么？', '你还记得我叫什么吗？我喜欢什么？', '我叫什么？喜欢什么？'];
for (const [n, k] of novelPairs(LIKE, 90))
  push('C_paraphrase', 'all', [n, k],
    [T(`我的名字叫${n}，我喜欢${k}。请记住。`), T(REPHRASE[(rnd() * REPHRASE.length) | 0], false)]);

// D. 干扰后回忆（中间插无关轮次）
for (const [n, k] of novelPairs(LIKE, 90))
  push('D_distractor', 'all', [n, k],
    [T(`我的名字叫${n}，我喜欢${k}。请记住。`), T('今天天气怎么样？', false), T('1加1等于多少？', false), T('我叫什么名字？喜欢什么？', false)]);

// E. 列表复制（顺序必须保留）
for (let i = 0; i < 60; i++) {
  const k = 3 + ((rnd() * 2) | 0);            // 3 或 4 项
  const pick = [], used = new Set();
  while (pick.length < k) { const x = ITEM[(rnd() * ITEM.length) | 0]; if (!used.has(x)) { used.add(x); pick.push(x); } }
  const want = pick.join('、');
  if (corpusSet.has(norm(`请按顺序输出：${want}。`))) continue;   // 顺序组合已见则跳过
  push('E_list', 'order', pick, [T(`请按顺序输出：${want}。不要添加其他文字。`)]);
}

// F. 原样复制（用词表重组的句子，语料中未见）
for (let i = 0; i < 60 && repeatSent.length; i++) {
  const base = repeatSent[(rnd() * repeatSent.length) | 0];
  const cand = base.replace(/^([\u4e00-\u9fa5]{2,4})/, NAME[(rnd() * NAME.length) | 0]);
  if (corpusSet.has(norm(`请重复这句话：${cand}。`))) continue;
  push('F_repeat', 'all', [cand], [T(`请重复这句话：${cand}。`)]);
}

// G. 算术（全新实例，语料中未出现的算式）
const ops = [['+', (a, b) => a + b], ['-', (a, b) => a - b], ['×', (a, b) => a * b]];
for (let i = 0; i < 60; i++) {
  const [op, f] = ops[(rnd() * ops.length) | 0];
  const a = 2 + ((rnd() * 20) | 0), b = 2 + ((rnd() * 12) | 0);
  const expr = `${a}${op}${b}`;
  if (raw.includes(expr)) continue;               // 该算式在语料中出现过则跳过
  push('G_arith', 'all', [String(f(a, b))], [T(`只输出数字：${expr}等于多少？`)]);
}

// ---------- 3) 泄漏审查（硬门禁） ----------
const leaks = [];
let checked = 0;
for (const it of items)
  for (const turn of it.turns)
    if (turn.novel) { checked++; if (corpusSet.has(norm(turn.t))) leaks.push({ id: it.cat, turn: turn.t }); }
if (leaks.length) {
  console.error(`泄漏审查失败：${leaks.length} 条"必须新颖"的话轮出现在训练语料中`);
  leaks.slice(0, 10).forEach(l => console.error('  [' + l.id + '] ' + l.turn));
  process.exit(1);
}
// 组合级审查：确认 A/B/C/D 用的配对确实是语料中未出现的
const pairLeak = items.filter(i => /^[ABCD]_/.test(i.cat) && seenPair.has(i.expect[0] + '|' + i.expect[1]));
if (pairLeak.length) { console.error(`组合泄漏：${pairLeak.length} 个配对已在语料中`); process.exit(1); }

// ---------- 4) 写出 ----------
fs.mkdirSync(OUT, { recursive: true });
const rows = items.map((it, i) =>
  [it.cat, 'I' + String(i + 1).padStart(4, '0'), it.mode, it.expect.join(';'), ...it.turns.map(t => t.t)].join('\t'));
const header = `# bench_v1 生成基准 —— 由 scripts/gen_bench.mjs 生成，请勿手改
# 列: CATEGORY \t ID \t MODE(all|order) \t EXPECT(;-sep) \t TURN1 \t TURN2 \t ...
# 语料: ${CORPUS}
# 归一化话轮数(去重): ${corpusSet.size}
# 词表规模: 名字 ${NAME.length}  喜好 ${LIKE.length}(颜色 ${COLOR.length})  列表项 ${ITEM.length}
# 语料中已见组合: ${seenPair.size}   本基准使用的未见组合(去重): ${new Set(items.filter(i=>/^[ABCD]_/.test(i.cat)).map(i=>i.expect[0]+'|'+i.expect[1])).size}
# 泄漏审查: 逐条话轮 0 命中；组合级 0 命中`;
fs.writeFileSync(OUT + '/spec.tsv', header + '\n' + rows.join('\n') + '\n');

const byCat = {};
for (const it of items) byCat[it.cat] = (byCat[it.cat] || 0) + 1;
console.log('基准已生成: ' + OUT + '/spec.tsv');
console.log('总条目 ' + items.length);
Object.entries(byCat).sort().forEach(([k, v]) => console.log('  ' + k.padEnd(14) + v));
console.log('\n泄漏审查: 必须新颖的话轮 ' + checked + ' 条，0 命中；配对 ' + items.filter(i => /^[ABCD]_/.test(i.cat)).length + ' 条，0 命中');
console.log('词表: 名字 ' + NAME.length + '，喜好 ' + LIKE.length + '（颜色 ' + COLOR.length + '），列表项 ' + ITEM.length);
