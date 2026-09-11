// 复制保真度：精确匹配（0/12）会掩盖「抄对一部分」的差异。
// 度量注入串与回答的字符级重叠，并用随机对照估计偶然水平。
// 用法: TAO_CPU_EXE=<exe> node scripts/eval_copy_fidelity.mjs <label>=<dsb> [...]
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const EXE = process.env.TAO_CPU_EXE || 'D:/TaoVm/build/yaoyao_cpu_v01.exe';
const CORPUS = process.env.TAO_CORPUS || 'D:/TaoVm/data/noffn_l4/conversations.txt';
const COLORS = ['紫色','蓝色','红色','黄色','橙色','棕色','青色','绿色','白色','粉色','灰色','金色'];

const SUR = '赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮齐康伍余元卜顾孟平黄和穆萧尹湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭林徐高夏蔡田樊胡凌霍虞万支柯昝管卢莫';
const GIV = ['伟','芳','娜','秀英','敏','静','丽','强','磊','洋','艳','勇','军','杰','娟','涛','明','超','秀兰','霞','平','刚','桂英','建国','文','辉','华','慧','燕','鹏','飞','宇','浩','晨','雪','琳','佳','婷','悦','辰','鑫','博','凯','宁','欣','怡','然','嘉','梦','雅','子涵','雨泽','梓萱','思远','若曦','明轩','子墨','一诺','浩然','诗涵','俊杰','晓东','志强','国华','春花','秋月','冬梅','小刚','小明','小红','小军','小丽','小华','小燕','小峰','小云','小龙','小凤','小鹏','小虎'];

function heldout(n) {
  const corpus = fs.readFileSync(CORPUS, 'utf8');
  let seed = 987654321;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const out = [];
  for (let i = 0; i < 20000 && out.length < n; i++) {
    const x = SUR[Math.floor(rnd() * SUR.length)] + GIV[Math.floor(rnd() * GIV.length)];
    if (corpus.includes(x) || out.includes(x)) continue;
    out.push(x);
  }
  return out;
}

// 最长公共子串长度（字符级）——「抄了多少」的直接度量
function lcs(a, b) {
  let best = 0;
  const m = a.length, n = b.length;
  let prev = new Int32Array(n + 1), cur = new Int32Array(n + 1);
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : 0;
      if (cur[j] > best) best = cur[j];
    }
    [prev, cur] = [cur, prev]; cur.fill(0);
  }
  return best;
}

const NAMES = heldout(12);
const tests = NAMES.map((n, i) => ({n, c: COLORS[i % COLORS.length]}));
const L = [];
for (const t of tests) L.push('/reset', `我的名字叫${t.n}，我喜欢${t.c}。请记住。`, '我叫什么名字？喜欢什么？');
L.push('/quit');
const input = L.join('\n') + '\n';

function replies(dsb) {
  const r = spawnSync(EXE, [dsb], {cwd: 'D:/TaoVm', input, encoding: 'utf8', timeout: 600000, maxBuffer: 1 << 26});
  const out = []; let cur = null;
  for (const ln of String(r.stdout || '').split(/\r?\n/)) {
    if (ln.startsWith('BEGIN_REPLY ')) { cur = []; continue; }
    if (ln.startsWith('END_REPLY ')) { if (cur) out.push(cur.join('')); cur = null; continue; }
    if (cur && ln !== 'RESET') cur.push(ln);
  }
  return out;
}

// 随机对照：把「回答」与**另一个**测试的名字配对，估计偶然的 LCS 水平
console.log(`held-out 名字: ${NAMES.join(' ')}\n`);
for (const arg of process.argv.slice(2)) {
  const eq = arg.indexOf('='); const label = arg.slice(0, eq), dsb = arg.slice(eq + 1);
  const reps = replies(dsb);
  let surnameHit = 0, givHit = 0, colorHit = 0, exact = 0;
  let nameLcs = 0, colorLcs = 0, ctrlLcs = 0;
  const rows = [];
  for (let i = 0; i < tests.length; i++) {
    const a = reps[2 * i + 1] || '';
    const t = tests[i];
    const sur = t.n.slice(0, 1), giv = t.n.slice(1);
    const sH = a.includes(sur), gH = a.includes(giv), cH = a.includes(t.c);
    const ex = a.includes(t.n) && cH;
    surnameHit += sH; givHit += gH; colorHit += cH; exact += ex;
    const nl = lcs(t.n, a);
    nameLcs += nl;
    colorLcs += lcs(t.c, a);
    // 对照：与下一个测试的名字比（同样分布、但无关）
    ctrlLcs += lcs(tests[(i + 1) % tests.length].n, a);
    rows.push(`  ${ex ? 'EXACT' : sH || gH ? 'partial' : '   -  '} 姓${sH ? 'Y' : '.'} 名${gH ? 'Y' : '.'} 色${cH ? 'Y' : '.'}  LCS名=${nl}  ${t.n}/${t.c} -> ${JSON.stringify(a).slice(0, 40)}`);
  }
  const N = tests.length;
  console.log(`########## ${label} ##########`);
  console.log(rows.join('\n'));
  console.log(`  精确 ${exact}/${N} | 姓氏命中 ${surnameHit}/${N} | 名字命中 ${givHit}/${N} | 颜色命中 ${colorHit}/${N}`);
  console.log(`  LCS(名字,回答) 均值 ${(nameLcs / N).toFixed(2)}   LCS(颜色,回答) 均值 ${(colorLcs / N).toFixed(2)}   随机对照 ${(ctrlLcs / N).toFixed(2)}\n`);
}
