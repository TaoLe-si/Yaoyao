// L5 判据评测（doc 17 §1.4）：核心是「多问合一」是否修好。
// 同一批 held-out 名字，四种问法，各自统计名字/颜色/两者。
// 用法: TAO_CPU_EXE=<exe> node scripts/eval_l5_criteria.mjs <label>=<dsb> [...]
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const EXE = process.env.TAO_CPU_EXE || 'D:/TaoVm/build/h2r_cpu.exe';
const CORPUS = process.env.TAO_CORPUS || 'D:/TaoVm/data/noffn_l5/conversations.txt';
const COLORS = ['紫色','蓝色','红色','黄色','橙色','棕色','青色','绿色','白色','粉色','灰色','金色'];

const SUR = '赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮齐康伍余元卜顾孟平黄和穆萧尹湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭林徐高夏蔡田樊胡凌霍虞万支柯昝管卢莫';
const GIV = ['伟','芳','娜','秀英','敏','静','丽','强','磊','洋','艳','勇','军','杰','娟','涛','明','超','秀兰','霞','平','刚','桂英','建国','文','辉','华','慧','燕','鹏','飞','宇','浩','晨','雪','琳','佳','婷','悦','辰','鑫','博','凯','宁','欣','怡','然','嘉','梦','雅','子涵','雨泽','梓萱','思远','若曦','明轩','子墨','一诺','浩然','诗涵','俊杰','晓东','志强','国华','春花','秋月','冬梅','小刚','小明','小红','小军','小丽','小华','小燕','小峰','小云','小龙','小凤','小鹏','小虎'];

function heldout(n) {
  const corpus = fs.readFileSync(CORPUS, 'utf8');
  let seed = 987654321;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const out = [];
  for (let i = 0; i < 30000 && out.length < n; i++) {
    const x = SUR[Math.floor(rnd() * SUR.length)] + GIV[Math.floor(rnd() * GIV.length)];
    if (corpus.includes(x) || out.includes(x)) continue;
    out.push(x);
  }
  return out;
}

const NAMES = heldout(12);
const tests = NAMES.map((n, i) => ({n, c: COLORS[i % COLORS.length]}));

// 四种问法；前两种是 L5 的核心修复目标
const FORMS = [
  ['两问合一',      '我叫什么名字？喜欢什么？'],
  ['两问合一(变体)', '请说出我的名字和喜好。'],
  ['只问名字',      '我叫什么名字？'],
  ['只问喜好',      '我喜欢什么？'],
];

const L = [];
for (const [, q] of FORMS) for (const t of tests) L.push('/reset', `我的名字叫${t.n}，我喜欢${t.c}。请记住。`, q);
L.push('/quit');
const input = L.join('\n') + '\n';

// 列表复制 + 原样复制（判据后两项）
const LISTS = [['红色','猫','桌子'], ['蓝色','狗','椅子','北京'], ['绿色','鱼','书包','尺子','上海']];
const NOVEL = ['小张在打球。','老师在备课。','妹妹在跳舞。','爷爷在下棋。','同学在画画。'];
const L2 = [];
for (const a of LISTS) L2.push('/reset', `请按顺序输出：${a.join('、')}。不要添加其他文字。`);
for (const s of NOVEL) L2.push('/reset', `请重复这句话：${s}`);
L2.push('/quit');

function replies(dsb, inp) {
  const r = spawnSync(EXE, [dsb], {cwd: 'D:/TaoVm', input: inp, encoding: 'utf8', timeout: 900000, maxBuffer: 1 << 26});
  const out = []; let cur = null;
  for (const ln of String(r.stdout || '').split(/\r?\n/)) {
    if (ln.startsWith('BEGIN_REPLY ')) { cur = []; continue; }
    if (ln.startsWith('END_REPLY ')) { if (cur) out.push(cur.join('')); cur = null; continue; }
    if (cur && ln !== 'RESET') cur.push(ln);
  }
  return out;
}

console.log(`held-out 名字（${CORPUS.split('/').slice(-2)[0]} 中不存在）: ${NAMES.join(' ')}\n`);
const summary = [];
for (const arg of process.argv.slice(2)) {
  const eq = arg.indexOf('='); const label = arg.slice(0, eq), dsb = arg.slice(eq + 1);
  const reps = replies(dsb, input);
  console.log(`########## ${label} ##########`);
  const res = {};
  FORMS.forEach(([fname], fi) => {
    let nk = 0, ck = 0, both = 0; const ex = [];
    for (let i = 0; i < tests.length; i++) {
      const a = reps[2 * (fi * tests.length + i) + 1] || '';
      const h1 = a.includes(tests[i].n), h2 = a.includes(tests[i].c);
      nk += h1; ck += h2; both += (h1 && h2);
      if (i < 2) ex.push(JSON.stringify(a).slice(0, 24));
    }
    res[fname] = {nk, ck, both};
    console.log(`  ${fname.padEnd(15)} 名字 ${String(nk).padStart(2)}/12  颜色 ${String(ck).padStart(2)}/12  两者 ${String(both).padStart(2)}/12   ${ex.join(' ')}`);
  });
  // 复制
  const r2 = replies(dsb, L2.join('\n') + '\n');
  let li = 0;
  LISTS.forEach((a, i) => { const g = r2[i] || '';
    if (a.every((x, k, arr) => g.includes(x) && (k === 0 || g.indexOf(x) > g.indexOf(arr[k - 1])))) li++; });
  let nov = 0;
  NOVEL.forEach((s, i) => { if ((r2[LISTS.length + i] || '').includes(s)) nov++; });
  console.log(`  列表复制 ${li}/3   原样复制(语料外) ${nov}/5`);
  summary.push({label, ...res, li, nov});
}
console.log('\n===== 判据对照（doc 17 §1.4）=====');
console.log('目标: 两问合一 名字≥9  颜色≥8  两者≥8 | 单问不劣化 | 列表≥2/3 | 复制≥1/5');
for (const s of summary) {
  const a = s['两问合一'], b = s['只问名字'], c = s['只问喜好'];
  console.log(`${s.label}\t两问合一 名${a.nk}/色${a.ck}/两${a.both}\t单问名${b.nk}/色${c.ck}\t列表${s.li}/3\t复制${s.nov}/5`);
}
