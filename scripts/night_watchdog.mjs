// 看门狗（两层）：① 大语料未启动且第3轮已结束超10分钟 → 接管启动；② 训练停滞 → 崩溃恢复。
// 全部判据都以"进程/时间"为准，避免把"慢"误判成"死"。
import {spawn, spawnSync} from 'node:child_process';
import fs from 'node:fs';
const BUILD='D:/TaoVm/build', STATUS=BUILD+'/night_status.txt', EXE='D:/TaoVm/build/train_shards.exe';
const STEPS=1000;   // 与 data/stage2_reason 一致（13维基+4真实推理+8alpaca末尾）；START_SHARD 固定 0，崩溃重启从头重跑，最终顺序仍正确
const STALE_MIN=15, TAKEover_MIN=10;
const log=(s)=>{const l=new Date().toISOString().slice(11,19)+'  '+s;console.log(l);try{fs.appendFileSync(STATUS,l+'\n','utf8');}catch(e){}};
const sleep=(ms)=>new Promise(r=>setTimeout(r,ms));
const R3 = BUILD + '/s1_s8r3/final.dsb';

const s2dirs=()=>fs.readdirSync(BUILD).filter(d=>d.startsWith('s2_night')&&fs.statSync(BUILD+'/'+d).isDirectory());
const anyDone=()=>s2dirs().some(d=>fs.existsSync(BUILD+'/'+d+'/final.dsb'));
const active=()=>{ let best=null;
  for (const d of s2dirs()) { const os=BUILD+'/'+d+'/opt_state.bin'; if (!fs.existsSync(os)) continue;
    if (fs.existsSync(BUILD+'/'+d+'/final.dsb')) continue;
    const m=fs.statSync(os).mtimeMs; if (!best||m>best.m) best={d,m}; } return best?best.d:null; };
const trainerAlive=()=>{ try { const r=spawnSync('tasklist',['/FI','IMAGENAME eq train_shards.exe','/NH'],{encoding:'utf8',timeout:20000});
    return String(r.stdout||'').toLowerCase().includes('train_shards'); } catch(e) { return null; } };
const launch=(out, resume, seed)=>{
  const args=['D:/TaoVm/data/stage2_reason','D:/TaoVm/build/tok_real_v1.bbp',out,String(STEPS),'32','32',resume,'0'];
  log('LAUNCH ' + args.join(' '));
  const p=spawn(EXE,args,{cwd:'D:/TaoVm',detached:true,stdio:'ignore',windowsHide:true,
    env:{...process.env,TAO_LR:'0.00005',TAO_SHUFFLE_SEED:String(seed),TAO_ALLOW_TOKENIZER:'1'}});
  p.unref(); log('LAUNCH_PID='+p.pid); };

let restarts=0, r3DoneSince=null, tookOver=false;
log('WATCHDOG_START stale=' + STALE_MIN + 'min launchTimeout=' + TAKEover_MIN + 'min');
for (let i=0;i<1700;i++) {
  await sleep(30000);
  if (anyDone()) { log('WATCHDOG_DONE'); break; }
  const d = active();
  if (!d) {
    // ---- 第一层：大语料根本没启动 ----
    if (fs.existsSync(R3)) {
      if (r3DoneSince === null) { r3DoneSince = Date.now(); log('R3_DONE_DETECTED 等待启动器'); continue; }
      const waited = (Date.now()-r3DoneSince)/60000;
      if (!tookOver && waited > TAKEover_MIN) {
        tookOver = true;
        const resume = fs.existsSync(R3.replace('/final.dsb','/opt_state.bin')) ? BUILD+'/s1_s8r3'
                     : (fs.existsSync(BUILD+'/s1_s8r2/opt_state.bin') ? BUILD+'/s1_s8r2' : null);
        if (resume && !fs.existsSync(BUILD+'/s2_night1')) { log('WATCHDOG_TAKEOVER 启动器未启动(等' + waited.toFixed(0) + 'min), 由看门狗接管'); launch(BUILD+'/s2_night1', resume, 20260920); }
        else log('WATCHDOG_TAKEOVER 跳过 (resume=' + resume + ')');
      }
    }
    continue;
  }
  r3DoneSince = null;
  // ---- 第二层：训练停滞 → 崩溃恢复 ----
  const age = (Date.now() - fs.statSync(BUILD+'/'+d+'/opt_state.bin').mtimeMs)/60000;
  if (age <= STALE_MIN) continue;
  const al = trainerAlive();
  if (al === true) { log('WATCHDOG_SLOW ' + d + ' stale=' + age.toFixed(1) + 'min 但进程仍在, 不重启'); continue; }
  if (restarts >= 5) { log('WATCHDOG_GIVEUP'); break; }
  restarts++;
  const out = BUILD + '/s2_night_r' + restarts;
  if (fs.existsSync(out)) { log('WATCHDOG skip existing ' + out); continue; }
  log('WATCHDOG_RESTART#' + restarts + ' stale=' + age.toFixed(1) + 'min resume=' + d);
  launch(out, BUILD+'/'+d, 20260920+restarts);
}
log('WATCHDOG_EXIT');
