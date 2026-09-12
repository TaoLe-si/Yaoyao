// 兜底报告：仅当 build/MORNING_REPORT.md 不存在时才写（幂等，绝不覆盖正式报告）。
// 由 Windows 计划任务在 morning_test 的 2 小时超时窗口之后执行。
import fs from 'node:fs';
const BUILD = 'D:/TaoVm/build';
const OUT = BUILD + '/MORNING_REPORT.md';
const log = s => { const l = new Date().toISOString().slice(11,19) + '  ' + s; console.log(l);
  try { fs.appendFileSync(BUILD + '/night_status.txt', l + '\n', 'utf8'); } catch(e){} };

if (fs.existsSync(OUT)) { log('FALLBACK_SKIP 正式报告已存在'); process.exit(0); }

let ldir = '', lstep = 0;
for (const d of fs.readdirSync(BUILD).filter(x => x.startsWith('s2_night') && fs.statSync(BUILD + '/' + x).isDirectory()))
  for (const e of fs.readdirSync(BUILD + '/' + d))
    if (e.startsWith('step_') && fs.existsSync(BUILD + '/' + d + '/' + e + '/final.dsb') && +e.slice(5) > lstep) { lstep = +e.slice(5); ldir = d; }

const ck = lstep ? 'build/' + ldir + '/step_' + lstep + '/final.dsb' : 'build/s2_wiki_phase/step_27659/final.dsb';
const B = String.fromCharCode(96).repeat(3);
const F = String.fromCharCode(96);
log('FALLBACK_WRITE 正式报告缺失, 最新检查点=' + ck);

fs.writeFileSync(OUT, [
  '# 明早测试报告（兜底版）',
  '',
  '生成时间: ' + new Date().toLocaleString('sv'),
  '',
  '> **[重要]** 正常评测流程没有产出报告（morning_test 未完成或超时）。',
  '> 本文件只保证**模型路径与测试命令可用**，**其中没有任何模型质量结论，请勿引用**。',
  '> 请在机器空闲时手动重跑下面的命令。',
  '',
  '**最新可用检查点**: ' + ck + (lstep ? '  (step ' + lstep + ')' : ''),
  '',
  '## 手动复测命令',
  '',
  B + 'powershell',
  'cd D:\\TaoVm',
  '$env:TAO_TOKENIZER="D:/TaoVm/build/tok_real_v1.bbp"; $env:TAO_CPU_THREADS="8"; $env:TAO_REP_WIN="0"',
  '',
  '# 真实能力（无重复惩罚）',
  '$env:TAO_REP_PEN="0"; node scripts\\eval_real.mjs ' + ck + ' data/alpaca_heldout.jsonl 400',
  '',
  '# 部署口径',
  '$env:TAO_REP_PEN="1.3"; node scripts\\eval_real.mjs ' + ck + ' data/alpaca_heldout.jsonl 400',
  '',
  '# 真实数学推理（Ape210K test 留出集）',
  'node scripts\\eval_math.mjs ' + ck + ' 100',
  '',
  '# 40 条完整问答',
  '$env:TAO_MODEL="D:/TaoVm/' + ck + '"; $env:TAO_PEN="1.3"; node scripts\\show_full.mjs',
  B,
  '',
  '## 对照基线',
  '',
  '- 精训最佳部署F1 **0.0658** @ ' + F + 'build/s1_s8r2/step_18424/final.dsb' + F,
  '- 精训最佳真实能力 **0.0479** @ ' + F + 'build/s1_s8r3/step_21624/final.dsb' + F,
  '- 数学推理基线 **1.0%** @ ' + F + 'build/s1_s8r3/step_21624/final.dsb' + F + '（退火后应显著高于此）',
  '',
  '详细说明见 ' + F + 'build/NIGHT_README.md' + F + '（含语料设计、故障处置、回退点）。',
  '',
].join('\n'), 'utf8');
log('FALLBACK_DONE -> ' + OUT);
