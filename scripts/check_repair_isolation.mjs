import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
process.chdir(path.join(path.dirname(fileURLToPath(import.meta.url)),'..'));
const required=['build/STOP_TRAINING','build/yaoyao_graph_step_1200.scp','build/yaoyao_graph_step_1200.dsb','build/yaoyao_graph_step_1204.scp','build/yaoyao_graph_step_1204.dsb'];
for(const f of required)if(!fs.existsSync(f))throw Error('missing safety artifact '+f);
if(fs.existsSync('build/graph_run.lock'))throw Error('production controller lock exists');
const state=JSON.parse(fs.readFileSync('build/graph_run_state.json','utf8'));
console.log(JSON.stringify({productionStopPresent:true,controllerLockAbsent:true,preservedCheckpoints:[1200,1204],statePhase:state.phase,note:'State phase may remain training after safe STOP; lock absence is not a complete OS process inventory. No mutation performed.'}));
