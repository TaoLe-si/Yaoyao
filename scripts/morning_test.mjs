// 明早一键测试：最新检查点 + 曲线最优检查点，双口径 n=400 + 40条完整问答，并附夜间事件。
// 用法: node scripts/morning_test.mjs [N]
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const N = +(process.argv[2] || 400);
const BUILD = 'D:/TaoVm/build';
const STATUS = BUILD + '/night_status.txt';
const log = (s) => { console.log(s); try{fs.appendFileSync(STATUS, new Date().toISOString().slice(11,19)+'  '+s+'\n','utf8');}catch(e){} };

const allCkpts = () => {
  const out = [];
  for (const d of fs.readdirSync(BUILD)) {
    if (!d.startsWith('s2_night') && !d.startsWith('s1_') && !d.startsWith('s2_conv')) continue;
    const dir = BUILD + '/' + d; if (!fs.statSync(dir).isDirectory()) continue;
    for (const e of fs.readdirSync(dir)) {
      if (!e.startsWith('step_')) continue;
      if (fs.existsSync(dir + '/' + e + '/final.dsb')) out.push({step:+e.slice(5), dsb:'build/' + d + '/' + e + '/final.dsb', dir:d});
    }
  }
  return out;
};
const list = allCkpts();
if (!list.length) { log('NO_CHECKPOINT_FOUND'); process.exit(1); }
const s2 = list.filter(c => c.dir.startsWith('s2_night') || c.dir.startsWith('s2_conv'));
const pool = (s2.length ? s2 : list).sort((a,b)=>b.step-a.step);
const latest = pool[0];

let best = null;
try {
  const lines = fs.readFileSync(BUILD+'/s2_progress.tsv','utf8').trim().split(/\r?\n/).slice(1);
  for (const l of lines) {
    const c = l.split('\t'); if (c.length < 9) continue;
    const f1 = parseFloat(c[7]); if (isNaN(f1)) continue;
    if (!best || f1 > best.f1) best = {step:+c[0], f1};
  }
} catch(e) {}
const bestCk = best ? pool.find(c => c.step === best.step) : null;
const bestInPool = !!bestCk;   // false = 曲线最优步属于已归档的旧轮（如维基轮），当前池中没有该检查点
log('TARGET latest step=' + latest.step + ' (' + latest.dir + ')'
  + (best ? '  curveBest@' + best.step + '=' + best.f1 + (bestInPool ? '' : ' [该步属已归档旧轮,不在池中]') : ''));

const run = (args, env) => spawnSync('node', args, {cwd:'D:/TaoVm', encoding:'utf8', maxBuffer:1<<26, timeout:7200000,
  env:{...process.env, TAO_TOKENIZER:'D:/TaoVm/build/tok_real_v1.bbp', TAO_CPU_THREADS:'8', TAO_REP_WIN:'0', ...env}});

const section = (ck, label) => {
  const o = ['## ' + label + ': ' + ck.dsb + '  (step ' + ck.step + ')\n'];
  for (const pen of ['0','1.3']) {
    const r = run(['scripts/eval_real.mjs', ck.dsb, 'data/alpaca_heldout.jsonl', String(N)], {TAO_REP_PEN:pen});
    o.push('### 定量 n=' + N + '  rep_pen=' + pen + '\n\n\x60\x60\x60\n' + String(r.stdout||'') + String(r.stderr||'') + '\x60\x60\x60\n');
  }
  const f = run(['scripts/show_full.mjs'], {TAO_MODEL:ck.dsb, TAO_PEN:'1.3', TAO_REP_PEN:'1.3'});
  o.push('### 40 条完整问答 (pi=1.3)\n\n' + String(f.stdout||'') + String(f.stderr||'') + '\n');
  const MN = Math.min(N, 100);
  const m = run(['scripts/eval_math.mjs', ck.dsb, String(MN)], {TAO_REP_PEN:'1.0'});
  o.push('### 真实数学推理 (Ape210K test 留出集, n=' + MN + ', 从未参与训练)\n\n'
    + '> 对照：推理训练前 s1_s8r3/step_21624 = **1.0%**；维基轮中期 step_28100 = **0.0%**\n\n'
    + '\x60\x60\x60\n' + String(m.stdout||'') + String(m.stderr||'') + '\x60\x60\x60\n');
  return o.join('\n');
};

let body = section(latest, '最新检查点');
if (bestCk && bestCk.step !== latest.step) {
  log('ALSO_BEST step=' + bestCk.step);
  body += '\n---\n\n' + section(bestCk, '抽样曲线中部署F1最优的检查点');
} else {
  body += '\n> ' + (best && !bestInPool
    ? '抽样曲线的最优点（step ' + best.step + '，部署F1 ' + best.f1 + '）属于**已归档的旧轮**（维基轮，问答退化），其检查点已移出扫描范围；上面的最新检查点即当前语料下的交付模型。'
    : '最新检查点即为抽样曲线中最优点（或曲线尚无数据）。') + '\n';
}

let ev = [];
try { ev = fs.readFileSync(STATUS,'utf8').split(/\r?\n/).filter(l=>/WARNING|RESTART|FAIL|ABORT|GIVEUP|UNCONFIRMED/.test(l)); } catch(e) {}
const s2dirs = fs.readdirSync(BUILD).filter(d=>d.startsWith('s2_night')&&fs.statSync(BUILD+'/'+d).isDirectory());

const head = '# 明早测试报告\n\n生成时间: ' + new Date().toLocaleString('sv')
  + '\n\n**最新检查点**: ' + latest.dsb + ' (step ' + latest.step + ')'
  + (best ? '\n\n**曲线最优**: step ' + best.step + ' (部署F1 ' + best.f1 + ')' + (bestInPool ? '' : '  ← 属已归档旧轮，当前池中无此检查点') : '')
  + '\n\n对比基准（精训）：部署F1 最高 0.0658 @ s1_s8r2/step_18424；单字循环最低 9.5% @ step_17050'
  + '\n\n### 夜间事件\n\n' + (ev.length ? ev.map(l=>'- ' + l.trim()).join('\n') : '- 无异常（未崩溃重启，未触发回退告警）')
  + '\n\n收敛门控轮(s2_conv*): ' + fs.readdirSync(BUILD).filter(x=>x.startsWith('s2_conv')&&fs.statSync(BUILD+'/'+x).isDirectory()).join(', ')
  + '\n\n阶段2 目录: ' + (s2dirs.length ? s2dirs.join(', ') : '(无)') + (s2dirs.length > 1 ? '   [注意] 多个目录 = 发生过崩溃自动重启' : '')
  + '\n\n---\n';

fs.writeFileSync(BUILD + '/MORNING_REPORT.md', head + '\n' + body, 'utf8');
log('REPORT -> build/MORNING_REPORT.md');
