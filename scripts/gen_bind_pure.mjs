// 纯 binding 受控实验语料。目的：判断"跨轮记住一个变量"这个能力，
// 在没有任何其他任务干扰、且名字不重复（无法靠记忆）的情况下能否学会。
// 训练名与评测名完全不相交 —— 所以考的是绑定，不是背诵。
import fs from 'node:fs';

const OUT = process.argv[2] || 'D:/TaoVm/data/bind_pure';
const HELD = process.argv[3] || 'D:/TaoVm/data/bind_heldout.txt';
const N_TRAIN = Number(process.argv[4] || 2400);
const N_HELD = Number(process.argv[5] || 120);

const XING = '赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳'.split('');
const MING = '伟芳娜秀英敏静丽强磊洋艳勇军杰娟涛明超霞平刚桂香文辉力鹏华健俊峰宇泽晨欣怡佳琳鑫博思远航宁康乐然安泰瑞祥丰春夏秋冬山川河海云雨雪霜'.split('');
const HOBBY = ['豆浆','草莓','桃子','铅笔','地图','自行车','电影','音乐','绿茶','面包','苹果','跑步','围棋','油画','风筝','橙汁','面条','葡萄','小提琴','登山','咖啡','西瓜','篮球','书法','钢琴','钓鱼','象棋','酸奶','橘子','游泳'];

let seed = 20260925;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

function makeNames(n) {
  const set = new Set();
  while (set.size < n) {
    const x = XING[Math.floor(rnd() * XING.length)];
    const len = 1 + Math.floor(rnd() * 2);
    let m = '';
    for (let i = 0; i < len; i++) m += MING[Math.floor(rnd() * MING.length)];
    set.add(x + m);
  }
  return [...set];
}

function doc(name, hobby, style) {
  if (style === 0) return ['U 我的名字叫' + name + '，我喜欢' + hobby + '。请记住。', 'A 记住了。', 'U 我叫什么名字？', 'A ' + name + '。'];
  if (style === 1) return ['U 请记住我的名字：' + name + '。', 'A 好的，我记住了。', 'U 我叫什么名字？', 'A ' + name + '。'];
  return ['U 我叫' + name + '，喜欢' + hobby + '，请记住。', 'A 记住了。', 'U 你还记得我的名字吗？', 'A 你叫' + name + '。'];
}

const all = makeNames(N_TRAIN + N_HELD);
const trainNames = all.slice(0, N_TRAIN);
const heldNames = all.slice(N_TRAIN);

fs.mkdirSync(OUT, { recursive: true });
const lines = [];
for (const n of trainNames) lines.push('DOC', ...doc(n, HOBBY[Math.floor(rnd() * HOBBY.length)], Math.floor(rnd() * 3)));
fs.writeFileSync(OUT + '/conversations.txt', lines.join('\n') + '\n');

const hl = [];
for (const n of heldNames) hl.push(JSON.stringify({ name: n, turns: doc(n, HOBBY[Math.floor(rnd() * HOBBY.length)], Math.floor(rnd() * 3)) }));
fs.writeFileSync(HELD, hl.join('\n') + '\n');

console.log('训练 docs=' + trainNames.length + '  评测 docs=' + heldNames.length);
console.log('名字交集=' + trainNames.filter(x => heldNames.includes(x)).length + ' (必须为 0)');
console.log('样例: ' + JSON.stringify(doc(trainNames[0], '豆浆', 0)));
