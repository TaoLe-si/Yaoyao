# Yaoyao runtime handoff

## Current run

Native process PID19232, executable build/train_yaoyao_parallel.exe, input build/yaoyao_continuous_step_33.scp, target233 (200 new updates). Log build/parallel_training_round47.log. Do not launch another trainer while this process exists. Last observed completed45; last confirmed durable40. Runtime process may advance after this note. Inspect log and process before action.

## Artifact meanings

- build/yaoyao_parallel_step_N.scp: master weights, Adam moments, step, four states and data cursors; resume with parallel trainer, not old100step driver.
- build/yaoyao_parallel_step_N.dsb: effective CPU inference weights; NEVER use as mastertrainingresume.
- build/formal_tokenizer.bbp: frozen16384vocab; tokenizerSHA25634463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333.
- Stop by creating build/STOP_TRAINING; wait for SAVED and STOP optimizer boundary and process exit. Do not removeSTOP before deliberate restart. Logs may be stale during active update.

## Next fixed evaluation

After SAVED step50 and model exists, run CPU evaluator once (no GPU contention):
```powershell
.\build\evaluate_yaoyao_cpu.exe build/yaoyao_parallel_step_50.dsb
```
Same23docs4906assistanttargets. Latest40CPU NLL8.909300780. Uniform9.704060528, fulltrainunigram8.811622545 (differentdataexposure). No successclaim. Avoid changing validationmask/tokenizer. Testsplitnotforselection.

## Recovery (only after process absence)

```powershell
.\build\train_yaoyao_parallel.exe build/yaoyao_parallel_step_50.scp --preflight
```
Replace50withlatestconfirmedcomplete checkpoint. Native invocation without --preflight starts200ADDITIONAL updates, not continuation tooldtarget233. For interrupted run, report newly chosen target explicitly; currentdriverdoesnotpersist originalrun target. Do not overwrite existingoutputfiles.

Legacy run_continuous_training.ps1 invokes older continuous executable and defaults31; it is NOT currentoptimizedresumeentry. Noautomaticlatestcheckpointdiscoveryimplemented.
