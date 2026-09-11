// pass 边界对照：同一 held-out 全量 400 条、双口径，供跨轮比较。
// 用法: node scripts/pass_ab.mjs <label=dsb> [...]
import { spawnSync } from 'node:child_process';
const args = process.argv.slice(2);
const CFG = [['0','0','真实能力'], ['1.3','0','部署口径']];
console.log('label\t口径\tloop\tF1\t乱码\t空回复\t长度比');
for (const a of args) {
  const k = a.indexOf('=');
  const label = a.slice(0, k), dsb = a.slice(k+1);
  for (const [pen, win, tag] of CFG) {
    const env = { ...process.env, TAO_TOKENIZER: 'D:/TaoVm/build/tok_real_v1.bbp', TAO_REP_PEN: pen, TAO_REP_WIN: win, TAO_CPU_THREADS: '8' };
    const p = spawnSync('node', ['D:/TaoVm/scripts/eval_real.mjs', dsb, 'D:/TaoVm/data/alpaca_heldout.jsonl', '400'],
      { cwd: 'D:/TaoVm', encoding: 'utf8', timeout: 1800000, maxBuffer: 1 << 27, env });
    const o = String(p.stdout || '');
    const g = k2 => { const m = o.match(new RegExp(k2 + '\\s*=\\s*([\\d.]+)')); return m ? m[1] : '?'; };
    console.log([label, tag, g('退化复读'), g('bigramF1'), g('乱码'), g('空回复'), g('长度比')].join('\t'));
  }
}
