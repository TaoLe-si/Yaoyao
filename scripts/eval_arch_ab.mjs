// 架构 A/B 评测：按预登记判据挑检查点，跑「交叉注入绑定率」+ held-out A–F 分类。
// 用法: node scripts/eval_arch_ab.mjs <label> <run_dir> <cpu_exe> <diag_exe> [NLL阈值]
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';

const [label, runDir, cpuExe, diagExe, thrArg] = process.argv.slice(2);
const THR = thrArg ? +thrArg : 0.3;
const log = runDir + '.log';
if (!fs.existsSync(log)) { console.error('缺少训练日志 ' + log); process.exit(2); }

// 1) 逐步 NLL
const steps = [];
for (const ln of fs.readFileSync(log, 'utf8').split(/\r?\n/)) {
  if (!ln.startsWith('UPDATE')) continue;
  const nll = ln.match(/train_preupdate_NLL=([\d.]+)/);
  const st = ln.match(/step=(\d+)/);
  if (nll && st) steps.push({step: +st[1], nll: +nll[1]});
}
if (!steps.length) { console.error('日志里没有 UPDATE 行'); process.exit(2); }

// 2) 已导出的检查点
const ckpts = fs.existsSync(runDir)
  ? fs.readdirSync(runDir).filter(d => /^step_\d+$/.test(d))
      .map(d => ({dir: d, step: +d.slice(5)})).sort((a, b) => a.step - b.step)
  : [];
const dsbOf = c => `${runDir}/${c.dir}/final.dsb`;

// 3) 预登记判据：首次 NLL<=THR 的步，取 >= 它的最小已导出检查点
const cross = steps.find(s => s.nll <= THR);
let pick = null, why = '';
if (cross) {
  pick = ckpts.find(c => c.step >= cross.step) || ckpts[ckpts.length - 1] || null;
  why = `首次 NLL<=${THR} 出现在 step ${cross.step}(NLL=${cross.nll})`;
} else {
  pick = ckpts[ckpts.length - 1] || null;
  why = `全程未达到 NLL<=${THR}（最低 ${Math.min(...steps.map(s => s.nll))}），取最后检查点`;
}
if (!pick) { console.error('没有可用检查点'); process.exit(2); }

const final = steps[steps.length - 1];
console.log(`\n########## ${label} ##########`);
console.log(`训练: ${steps.length} 步, 末步 NLL=${final.nll}, 最低 NLL=${Math.min(...steps.map(s => s.nll))}`);
console.log(`判据: ${why}`);
console.log(`选用: ${dsbOf(pick)} (step ${pick.step})`);

// 4) 绑定率
const d = spawnSync('D:/TaoVm/' + diagExe.replace(/^build\//, 'build\\'), [dsbOf(pick).replace(/\//g, '\\')],
  {cwd: 'D:/TaoVm', encoding: 'utf8', timeout: 600000, maxBuffer: 1 << 26});
const dtxt = String(d.stdout || '') + String(d.stderr || '');
const bind = dtxt.split(/\r?\n/).filter(l => /^总计|^名字在词表内|^名字不在词表内/.test(l));
console.log('\n--- 交叉注入绑定率 ---');
console.log(bind.length ? bind.join('\n') : dtxt.trim().slice(-400));

// 5) held-out 分类
const e = spawnSync('cmd.exe', ['/c',
  `node scripts\\eval_generalization.mjs ${label}=${dsbOf(pick).replace(/\//g, '/')}`],
  {cwd: 'D:/TaoVm', encoding: 'utf8', timeout: 600000, maxBuffer: 1 << 26,
   env: {...process.env, TAO_CPU_EXE: 'D:/TaoVm/' + cpuExe}});
const etxt = String(e.stdout || '') + String(e.stderr || '');
console.log('\n--- held-out A–F 分类 ---');
console.log(etxt.split(/\r?\n/).filter(l => /分类:|TOTAL:/.test(l)).join('\n') || etxt.trim().slice(-400));

console.log('\n--- 逐条明细 ---');
console.log(etxt.split(/\r?\n/).filter(l => /PASS|FAIL/.test(l)).join('\n'));
