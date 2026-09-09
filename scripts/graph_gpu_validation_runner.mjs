import fs from 'node:fs';
import {spawn} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
process.chdir(path.join(path.dirname(fileURLToPath(import.meta.url)),'..'));
const statePath='build/graph_run_state.json', lockPath='build/graph_run.lock';
const atomic=(p,s)=>{fs.writeFileSync(p+'.tmp',s,{flag:'wx'});fs.renameSync(p+'.tmp',p);};
let child=null;let currentLR=0;const seen=new Set(fs.readFileSync('build/loss_curve.csv','utf8').split(/\r?\n/).filter(x=>x.includes(',validation,')).map(x=>Number(x.split(',')[2])));
const lock=fs.openSync(lockPath,'wx');fs.writeSync(lock,String(process.pid));
const run=(exe,args,log)=>new Promise((resolve,reject)=>{const fd=fs.openSync(log,'wx');const fixed=fs.openSync('build/training.log','a');child=spawn(exe,args,{stdio:['ignore','pipe','pipe'],windowsHide:true});const stamp=JSON.stringify({time:new Date().toISOString(),pid:child.pid,exe,log});console.log(stamp);fs.writeSync(fixed,stamp+'\n');let pending='';const consume=d=>{fs.writeSync(fd,d);fs.writeSync(fixed,d);pending+=d.toString();let i;while((i=pending.indexOf('\n'))>=0){const line=pending.slice(0,i);pending=pending.slice(i+1);const v=line.match(/^FIXED_VALIDATION step=(\d+) backend=cuda-forward-arch2-effective-resident-v1 docs=23 supervised=4906 positions=\d+ NLL=([0-9.]+) dataset_sha256=69900a8acbb28b09296b13a08c76657aa6261c129e929d59bc83ad9d2f1447b0 tokenizer_sha256=34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333/);if(v&&Number(v[1])%100!==0&&!seen.has(Number(v[1]))){fs.appendFileSync('build/loss_curve.csv',`${new Date().toISOString()},validation,${v[1]},${v[2]},${currentLR}\n`);seen.add(Number(v[1]));}const m=line.match(/UPDATE step=(\d+).*train_preupdate_NLL=([0-9.]+).*lr=([0-9.e+-]+)/);if(m)fs.appendFileSync('build/loss_curve.csv',`${new Date().toISOString()},train,${m[1]},${m[2]},${m[3]}\n`);}};child.stdout.on('data',consume);child.stderr.on('data',consume);child.once('error',e=>{fs.closeSync(fd);fs.closeSync(fixed);child=null;reject(e);});child.once('exit',code=>{fs.closeSync(fd);fs.closeSync(fixed);child=null;code===0?resolve():reject(new Error('child exit '+code+'; inspect '+log));});});
try{
if(!fs.existsSync(statePath))throw Error('explicit graph resume state required');let s=JSON.parse(fs.readFileSync(statePath,'utf8'));
if(s.version!==1||!Number.isFinite(s.lr)||s.lr<0.000001||s.lr>0.001||s.target!==2000)throw Error('invalid controller state');
const persist=()=>atomic(statePath,JSON.stringify(s,null,2)+'\n');
while(s.phase!=='complete'){
if(fs.existsSync('build/STOP_TRAINING')){console.log('STOP requested; inspect latest saved trainer log before resume');break;}
if(s.phase==='ready'){
if(s.step>=s.target){s.phase='complete';persist();break;}
s.next=Math.min(s.target,(Math.floor(s.step/100)+1)*100);s.phase='training';persist();
}
currentLR=s.lr;const stem=s.next===60?'build/yaoyao_controlled_step_60':'build/yaoyao_graph_step_'+s.next;
if(s.phase==='training'){
if(!fs.existsSync(stem+'.scp')||!fs.existsSync(stem+'.dsb')){
if(fs.existsSync(stem+'.scp')||fs.existsSync(stem+'.dsb'))throw Error('partial checkpoint; manual inspection required');
atomic(s.checkpoint+'.control','TC1 '+s.lr+' '+s.next+' 0\n');
await run('build/train_yaoyao_graph_gpuval.exe',[s.checkpoint],'build/graph_train_'+s.next+'_'+Date.now()+'.log');
if(fs.existsSync('build/STOP_TRAINING'))break;
if(!fs.existsSync(stem+'.scp')||!fs.existsSync(stem+'.dsb'))throw Error('expected checkpoint missing');
}
const raw=fs.readFileSync(fs.readdirSync('build').filter(x=>x.startsWith('graph_train_'+s.next+'_')).sort().map(x=>'build/'+x).at(-1),'utf8');const metric=raw.split(/\r?\n/).find(x=>x.startsWith('FIXED_VALIDATION step='+s.next+' '));if(metric&&!fs.existsSync('build/graph_gpu_eval_'+s.next+'.log'))fs.writeFileSync('build/graph_gpu_eval_'+s.next+'.log',metric+'\n',{flag:'wx'});s.phase='evaluation';persist();
}
if(s.phase==='evaluation'){
const log='build/graph_gpu_eval_'+s.next+'.log';
if(!fs.existsSync(log))await run('build/train_yaoyao_graph_gpuval.exe',[stem+'.scp','--validate-only'],log);
const text=fs.readFileSync(log,'utf8');const match=text.match(/FIXED_VALIDATION step=(\d+) backend=cuda-forward-arch2-effective-resident-v1 docs=23 supervised=4906 positions=\d+ NLL=([0-9.]+) dataset_sha256=69900a8acbb28b09296b13a08c76657aa6261c129e929d59bc83ad9d2f1447b0 tokenizer_sha256=34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333/);
if(!match||Number(match[1])!==s.next)throw Error('evaluation evidence mismatch');
const nll=Number(match[2]);if(!Number.isFinite(nll)||nll<0||s.next<=s.lastEval)throw Error('invalid/repeated metric');
const dialogue='build/graph_dialogue_'+s.next+'.txt';
// Routine dialogue generation disabled at user request; fixed validation retained for LR/quality metric.
fs.appendFileSync('build/loss_curve.csv',`${new Date().toISOString()},validation,${s.next},${nll},${s.lr}\n`);s.lastEval=s.next;s.step=s.next;s.checkpoint=stem+'.scp';s.nll=nll;
if(nll<=2.5){s.phase='complete';s.reason='loss target reached; generation review required';}
else{if(s.best===null||nll<=s.best-.02){s.best=nll;s.stale=0;}else if(++s.stale>=3){s.lr=Math.max(.00003,s.lr*.5);s.stale=0;}s.phase=s.step>=s.target?'complete':'ready';}
persist();console.log(JSON.stringify({step:s.step,nll,lr:s.lr,stale:s.stale,phase:s.phase}));
}
}
}catch(e){console.error(e);process.exitCode=1;}finally{fs.closeSync(lock);fs.unlinkSync(lockPath);}
