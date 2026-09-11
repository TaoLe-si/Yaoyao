// 分布内绑定评测：用**与训练同分布、但语料中不存在**的名字，避免记忆混淆。
// 用法: node scripts/eval_binding_indist.mjs <label>=<dsb> [...]
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const EXE = process.env.TAO_CPU_EXE || 'D:/TaoVm/build/yaoyao_cpu_v01.exe';
const CORPUS = process.env.TAO_CORPUS || 'D:/TaoVm/data/noffn_l4/conversations.txt';
const COLORS = ['紫色','蓝色','红色','黄色','橙色','棕色','青色','绿色','白色','粉色','灰色','金色'];

// 与 data/noffn_l4 生成器相同的姓氏/名字表
const SUR = '赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮齐康伍余元卜顾孟平黄和穆萧尹湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭林徐高夏蔡田樊胡凌霍虞万支柯昝管卢莫';
const GIV = ['伟','芳','娜','秀英','敏','静','丽','强','磊','洋','艳','勇','军','杰','娟','涛','明','超','秀兰','霞','平','刚','桂英','建国','文','辉','华','慧','燕','鹏','飞','宇','浩','晨','雪','琳','佳','婷','悦','辰','鑫','博','凯','宁','欣','怡','然','嘉','梦','雅','子涵','雨泽','梓萱','思远','若曦','明轩','子墨','一诺','浩然','诗涵','俊杰','晓东','志强','国华','春花','秋月','冬梅','小刚','小明','小红','小军','小丽','小华','小燕','小峰','小云','小龙','小凤','小鹏','小虎'];

function heldoutNames(n) {
  const corpus = fs.readFileSync(CORPUS, 'utf8');
  let seed = 987654321;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const out = [];
  for (let i = 0; i < 20000 && out.length < n; i++) {
    const name = SUR[Math.floor(rnd() * SUR.length)] + GIV[Math.floor(rnd() * GIV.length)];
    if (corpus.includes(name) || out.includes(name)) continue;
    out.push(name);
  }
  return out;
}

function parseReplies(out) {
  const reps = []; let cur = null;
  for (const ln of out.split(/\r?\n/)) {
    if (ln.startsWith('BEGIN_REPLY ')) { cur = []; continue; }
    if (ln.startsWith('END_REPLY ')) { if (cur) reps.push(cur.join('')); cur = null; continue; }
    if (cur && ln !== 'RESET') cur.push(ln);
  }
  return reps;
}

function run(dsb, input) {
  const r = spawnSync(EXE, [dsb], {cwd: 'D:/TaoVm', input, encoding: 'utf8', timeout: 600000, maxBuffer: 1 << 26});
  return parseReplies(String(r.stdout || ''));
}

const NAMES = heldoutNames(12);
const tests = NAMES.map((n, i) => ({n, c: COLORS[i % COLORS.length]}));

const L = [];
for (const t of tests) L.push('/reset', `我的名字叫${t.n}，我喜欢${t.c}。请记住。`, '我叫什么名字？喜欢什么？');
L.push('/quit');
const input = L.join('\n') + '\n';

console.log(`分布内 held-out 名字（${CORPUS} 中不存在）: ${NAMES.join(' ')}\n`);
for (const arg of process.argv.slice(2)) {
  const eq = arg.indexOf('='); const label = arg.slice(0, eq), dsb = arg.slice(eq + 1);
  const reps = run(dsb, input);
  let nOK = 0, cOK = 0, both = 0, degen = 0;
  const rows = [];
  for (let i = 0; i < tests.length; i++) {
    const a = reps[2 * i + 1] || '(missing)';
    const t = tests[i];
    const nk = a.includes(t.n), ck = a.includes(t.c);
    nOK += nk; cOK += ck; both += (nk && ck);
    if (/(.)\1{3,}/.test(a)) degen++;
    rows.push(`  ${nk ? 'N' : '.'}${ck ? 'C' : '.'} ${t.n}/${t.c} -> ${JSON.stringify(a).slice(0, 46)}`);
  }
  console.log(`########## ${label} ##########`);
  console.log(rows.join('\n'));
  console.log(`  名字 ${nOK}/${tests.length}  颜色 ${cOK}/${tests.length}  两者 ${both}/${tests.length}  退化重复 ${degen}/${tests.length}\n`);
}
