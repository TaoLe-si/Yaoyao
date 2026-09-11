
// Held-out (out-of-corpus) verbatim-copy test — the decisive check for defect 3.
//
// Background: the original corpus had only 15 unique verbatim sentences x40 reps,
// so the model learned a 15-entry lookup table (in-corpus 6/6, out-of-corpus 0/5,
// and on novel input it emitted a DIFFERENT corpus sentence or repeated itself).
// rebalanced_v2 replaced that with 2615 unique sentences. This test asks whether
// the model now performs a genuine copy circuit on input it has never seen.
//
// Usage: node scripts/eval_verbatim_heldout.mjs <dsb> [corpus]
// Every generated sentence is verified ABSENT from the corpus before use.
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const DSB = process.argv[2];
const CORPUS = process.argv[3] || 'D:/TaoVm/data/rebalanced_v2/conversations.txt';
const EXE = process.env.TAO_CPU_EXE || 'D:/TaoVm/build/h2r_cpu.exe';
if (!DSB || !fs.existsSync(DSB)) { console.log('usage: node scripts/eval_verbatim_heldout.mjs <dsb> [corpus]'); process.exit(1); }

const PERSON=['弟弟','妹妹','哥哥','姐姐','爷爷','奶奶','爸爸','妈妈','叔叔','阿姨','小张','小王','小李','小明','小红','小刚','小丽','老师','同学','医生','护士','司机','工人','农民','学生','孩子','老人','邻居','表哥','表妹','舅舅','姑姑','诗人','画家','歌手','球员','教练','警察','消防员'];
const ANIMAL=['小猫','小狗','兔子','小鸟','金鱼','乌龟','熊猫','大象','猴子','老虎','狮子','蝴蝶','蜜蜂','松鼠','鸭子','公鸡','山羊','白马','绵羊','青蛙'];
const THING =['苹果','桃子','草莓','西瓜','葡萄','香蕉','橙子','书','铅笔','橡皮','书包','桌子','椅子','台灯','水杯','雨伞','地图','钟表','尺子','帽子','外套','鞋子','眼镜','手机','电脑','电视','自行车','汽车','火车','飞机','轮船','花瓶','窗帘','地毯','枕头','被子','碗','盘子','筷子','勺子'];
const PLACE =['公园','学校','医院','图书馆','超市','车站','操场','教室','厨房','阳台','花园','山顶','河边','海边','田野','森林','街道','广场','博物馆','电影院','动物园','菜市场','办公室','实验室','体育馆'];
const TIME  =['早上','中午','下午','晚上','今天','明天','昨天','周末','春天','夏天','秋天','冬天','清晨','傍晚','半夜','星期天'];
const ADV   =['认真','仔细','安静','开心','慢慢地','飞快地','悄悄地','高兴地','努力','一起','已经','正在','刚刚','常常','偶尔'];
const VERB  =['写字','看书','画画','唱歌','跳舞','跑步','做饭','下棋','喝茶','散步','睡觉','买菜','打球','备课','洗碗','浇花','写信','听音乐','看电视','整理房间'];

let seed=987654321;
const rnd=()=>{seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff;};
const P=(arr)=>arr[Math.floor(rnd()*arr.length)];

const TPL=[
  ()=>P(PERSON)+'在'+P(VERB)+'。',
  ()=>P(PERSON)+'和'+P(PERSON)+'一起去了'+P(PLACE)+'。',
  ()=>P(TIME)+'，'+P(PERSON)+'在'+P(PLACE)+'散步。',
  ()=>P(ANIMAL)+'在'+P(['吃草','睡觉','玩耍','晒太阳','找食物','游泳'])+'。',
  ()=>P(PLACE)+'里有很多'+P(THING)+'。',
  ()=>P(PERSON)+P(ADV)+'地打开门。',
  ()=>'桌子上有'+P(['一','两','三','五','七','九','十'])+'个'+P(THING)+'。',
  ()=>'我看见了'+P(THING)+'。',
  ()=>P(THING)+'和'+P(THING)+'都很好。',
  ()=>P(PERSON)+'把'+P(THING)+'放在了'+P(['桌上','椅子上','窗台上','门口','包里','书架上','地上'])+'。',
  ()=>P(ANIMAL)+'和'+P(ANIMAL)+'是好朋友。',
  ()=>'这'+P(['个','只','本','张','把'])+P(THING)+'是'+P(PERSON)+'的。',
  ()=>P(PERSON)+'一边走一边'+P(['说话','笑','想事情','点头'])+'。',
  ()=>'如果下雨，我们就去'+P(PLACE)+'。',
  ()=>P(PERSON)+'觉得'+P(THING)+'很好吃。',
];

const corpus=fs.readFileSync(CORPUS,'utf8');
const cand=[]; const used=new Set();
let guard=0;
while(cand.length<24 && guard<20000){
  guard++;
  const s=TPL[Math.floor(rnd()*TPL.length)]();
  if(!s || used.has(s)) continue;
  if(corpus.includes(s)) continue;   // strictly held out
  used.add(s); cand.push(s);
}
if(cand.length<8){ console.log('could not build enough held-out sentences'); process.exit(1); }

const inp=[];for(const s of cand)inp.push('/reset','请重复这句话：'+s);inp.push('/quit');
const r=spawnSync(EXE,[DSB],{cwd:'D:/TaoVm',input:inp.join('\n')+'\n',encoding:'utf8',timeout:1800000,maxBuffer:1<<26});
const reps=[];let c=null;
for(const ln of String(r.stdout||'').split(/\r?\n/)){
  if(ln.startsWith('BEGIN_REPLY ')){c=[];continue;}
  if(ln.startsWith('END_REPLY ')){if(c)reps.push(c.join(''));c=null;continue;}
  if(c&&ln!=='RESET')c.push(ln);
}
let ok=0;
console.log('===== 语料外「原样复制」测试（全部句子已验证不在语料中）=====');
cand.forEach((s,i)=>{
  const got=(reps[i]||'').trim();
  const good=got===s;
  if(good)ok++;
  console.log('  '+(good?'✓':'✗')+' 期望='+JSON.stringify(s)+'  模型='+JSON.stringify(got).slice(0,50));
});
console.log('');
console.log('  语料外原样复制: '+ok+'/'+cand.length);
