import {spawnSync} from 'node:child_process';
const MODEL=process.argv[2]||'build/s2_night1/step_50659/final.dsb';
// 单行提示：训练格式 U...A   vs 裸题目
const cases=[
 ['训练格式','U 一个超市购进5吨大米，5天卖出2000千克，还剩多少千克？ A '],
 ['训练格式','U 学校有75个篮球，35个排球，把这些球平均分给5个班，每个班分得几个球？ A '],
 ['训练格式','U 小红每分钟打110个字，她从10点开始打字，10点二十五分结束，共打了多少个字？ A '],
 ['训练格式','U 蔬菜市场运回茄子1200千克．运回的西红柿是茄子的(1/3)．西红柿有多少千克？ A '],
 ['裸题目','一个超市购进5吨大米，5天卖出2000千克，还剩多少千克？'],
 ['裸题目','学校有75个篮球，35个排球，把这些球平均分给5个班，每个班分得几个球？'],
];
const input=[];
for(const [k,p] of cases){ input.push('/reset'); input.push(p); }
input.push('/quit');
const r=spawnSync('D:/TaoVm/build/h2r_cpu.exe',[MODEL,'--rep-pen','1.2','--rep-win','0'],{cwd:'D:/TaoVm',input:input.join('\n')+'\n',encoding:'utf8',timeout:900000,maxBuffer:1<<27,
 env:{...process.env,TAO_TOKENIZER:'D:/TaoVm/build/tok_real_v1.bbp',TAO_CPU_THREADS:'6'}});
const b=[...String(r.stdout||'').matchAll(/BEGIN_REPLY \d+\r?\n([\s\S]*?)\r?\nEND_REPLY/g)].map(m=>m[1]);
cases.forEach(([k,p],i)=>{
 console.log('【'+k+'】'+p.replace(/^U /,'').replace(/ A $/,''));
 console.log('   → '+(b[i]||'(空)').replace(/\s+/g,' ').slice(0,240));
});
