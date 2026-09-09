import fs from 'node:fs';
import {spawn} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
process.chdir(path.join(path.dirname(fileURLToPath(import.meta.url)),'..'));
const statePath='build/controlled_run_state.json', lockPath='build/controlled_run.lock';
const atomic=(p,s)=>{fs.writeFileSync(p+'.tmp',s,{flag:'wx'});fs.renameSync(p+'.tmp',p);};
let child=null;
const lock=fs.openSync(lockPath,'wx');fs.writeSync(lock,String(process.pid));
const run=(exe,args,log)=>new Promise((resolve,reject)=>{const fd=fs.openSync(log,'wx');child=spawn(exe,args,{stdio:['ignore',fd,fd],windowsHide:true});console.log(JSON.stringify({pid:child.pid,exe,log}));child.once('error',e=>{fs.closeSync(fd);child=null;reject(e);});child.once('exit',code=>{fs.closeSync(fd);child=null;code===0?resolve():reject(new Error('child exit '+code+'; inspect '+log));});});
try{
let s=fs.existsSync(statePath)?JSON.parse(fs.readFileSync(statePath,'utf8')):{version:1,phase:'ready',step:24,checkpoint:'build/yaoyao_controlled_step_24.scp',target:224,lr:0.0005,best:null,stale:0,lastEval:0};
if(s.version!==1||!Number.isFinite(s.lr)||s.lr<0.000001||s.lr>0.001||s.target!==224)throw Error('invalid controller state');
const persist=()=>atomic(statePath,JSON.stringify(s,null,2)+'\n');
while(s.phase!=='complete'){
if(fs.existsSync('build/STOP_TRAINING')){console.log('STOP requested; inspect latest saved trainer log before resume');break;}
if(s.phase==='ready'){
if(s.step>=s.target){s.phase='complete';persist();break;}
s.next=Math.min(s.target,(Math.floor(s.step/10)+1)*10);s.phase='training';persist();
}
const stem='build/yaoyao_controlled_step_'+s.next;
if(s.phase==='training'){
if(!fs.existsSync(stem+'.scp')||!fs.existsSync(stem+'.dsb')){
if(fs.existsSync(stem+'.scp')||fs.existsSync(stem+'.dsb'))throw Error('partial checkpoint; manual inspection required');
atomic(s.checkpoint+'.control','TC1 '+s.lr+' '+s.next+' 0\n');
await run('build/train_yaoyao_controlled.exe',[s.checkpoint],'build/controlled_train_'+s.next+'_'+Date.now()+'.log');
if(fs.existsSync('build/STOP_TRAINING'))break;
if(!fs.existsSync(stem+'.scp')||!fs.existsSync(stem+'.dsb'))throw Error('expected checkpoint missing');
}
s.phase='evaluation';persist();
}
if(s.phase==='evaluation'){
const log='build/controlled_eval_'+s.next+'.log';
if(!fs.existsSync(log))await run('build/evaluate_arch2.exe',[stem+'.dsb'],log);
const text=fs.readFileSync(log,'utf8');const match=text.match(/VALIDATION backend=CPU docs=23 supervised=4906 NLL=([0-9.]+) model=(\S+)/);
if(!match||match[2]!==stem+'.dsb')throw Error('evaluation evidence mismatch');
const nll=Number(match[1]);if(!Number.isFinite(nll)||nll<0||s.next<=s.lastEval)throw Error('invalid/repeated metric');
const dialogue='build/controlled_dialogue_timed_avx2_'+s.next+'.txt';
if(!fs.existsSync(dialogue))await run('build/dialogue_avx2.exe',[stem+'.dsb'],dialogue);
s.lastEval=s.next;s.step=s.next;s.checkpoint=stem+'.scp';s.nll=nll;
if(nll<=2.5){s.phase='complete';s.reason='loss target reached; generation review required';}
else{if(s.best===null||nll<=s.best-.02){s.best=nll;s.stale=0;}else if(++s.stale>=3){s.lr=Math.max(.00003,s.lr*.5);s.stale=0;}s.phase=s.step>=s.target?'complete':'ready';}
persist();console.log(JSON.stringify({step:s.step,nll,lr:s.lr,stale:s.stale,phase:s.phase}));
}
}
}catch(e){console.error(e);process.exitCode=1;}finally{fs.closeSync(lock);fs.unlinkSync(lockPath);}
