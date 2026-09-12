// 夜间接力：等精训第3轮结束后，自动从最优检查点接入大语料(阶段2)训练。
// 训练进程以 detached 方式启动并 unref —— 即使本 Node 进程/作业被回收，训练也不会中断。
import {spawn} from 'node:child_process';
import fs from 'node:fs';

const EXE       = 'D:/TaoVm/build/train_shards.exe';
const SHARD_DIR = 'D:/TaoVm/data/stage2_night';
const TOK       = 'D:/TaoVm/build/tok_real_v1.bbp';
const OUT       = 'D:/TaoVm/build/s2_night1';
const LOG       = OUT + '.log';
const STEPS     = +fs.readFileSync('D:/TaoVm/build/stage2_steps.txt','utf8').trim();
const STATUS    = 'D:/TaoVm/build/night_status.txt';

const log = (s) => { const l = new Date().toISOString().slice(11,19) + '  ' + s; console.log(l); try{fs.appendFileSync(STATUS,l+'\n','utf8');}catch(e){} };
const ok  = (d) => fs.existsSync(d + '/final.dsb') && fs.existsSync(d + '/opt_state.bin');
const sleep = (ms) => new Promise(r=>setTimeout(r,ms));

log('WATCHER_START(detached) steps_per_shard=' + STEPS + ' shards=' + fs.readdirSync(SHARD_DIR).filter(f=>f.endsWith('.bin')).length);
if (fs.existsSync(OUT)) { log('ABORT: ' + OUT + ' 已存在'); process.exit(3); }

let resume = null;
for (let i = 0; i < 400; i++) {            // 最多等 200 分钟
  if (ok('D:/TaoVm/build/s1_s8r3')) { resume = 'D:/TaoVm/build/s1_s8r3'; break; }
  await sleep(30000);
}
if (!resume && ok('D:/TaoVm/build/s1_s8r2')) { resume = 'D:/TaoVm/build/s1_s8r2'; log('R3 未就绪, 退回 R2'); }
if (!resume) { log('NO_RESUME_AVAILABLE — 放弃启动'); process.exit(2); }
log('RESUME_FROM=' + resume);

const args = [SHARD_DIR, TOK, OUT, String(STEPS), '32', '32', resume, '0'];
const env  = {...process.env, TAO_LR: '0.0001', TAO_SHUFFLE_SEED: '20260920', TAO_ALLOW_TOKENIZER: '1'};
log('LAUNCH(detached) ' + args.join(' '));
const p = spawn(EXE, args, {cwd: 'D:/TaoVm', detached: true, stdio: 'ignore', windowsHide: true, env});
p.unref();
log('S2_PID=' + p.pid);

// 确认真的起来了：等日志出现 SHARD_TRAIN_START
let started = false;
for (let i = 0; i < 40; i++) {
  await sleep(5000);
  if (fs.existsSync(LOG)) {
    const t = fs.readFileSync(LOG).toString('utf8');
    if (t.includes('SHARD_TRAIN_START')) { started = true; log('S2_CONFIRMED ' + t.split(/\r?\n/).filter(l=>l.includes('SHARD_TRAIN_START'))[0].slice(0,110)); break; }
    if (t.includes('FAIL')) { log('S2_FAILED ' + t.slice(0,200)); break; }
  }
}
if (!started) log('S2_UNCONFIRMED — 请手动检查 ' + LOG);
log('WATCHER_EXIT');
