# Frozen-readout interference decomposition

Native diagnose_chain_interference.cpp reuses exact float training protocol from QAT comparison(no ternary run). Same seed initializations,1344train noise0/1,1200steps,24tanhreadout. TrainingCE and test predictions reproduce baseline. Four interventions at evaluation only on224heldoutgroup examples with3suffix distractor steps,then query normally processed. Same trained weights across interventions per seed,no refitting.

Accuracy seed42/123/2026:
normal permutation+write12.0536/14.2857/8.0357percent
permutation only13.8393/12.0536/9.8214
write only58.4821/47.7679/70.9821
freeze both100/100/100

Freeze state exactly equals corresponding no-distractor state224/224 eachseed. This is identity construction using ORACLE knowledge of distractor suffix,not learned invariance;returns to seen length0representation. It shows protected representation is usable,NOT ability to identify irrelevant input. Sum-only injection uses same three tokens at fixed coordinate then ordinary queryupdate;interventions alter state distribution,not additive attribution/independent causal percentages.

Conclusion:three extra untrained rotations alone suffice to break this fixed readout;content-only writes also degrade performance. Both position sensitivity and contamination present. Does NOT prove all recurrence needs freeze or global order unnecessary. Learning selective update and handling relevant facts within noise still untested. Test already reused;diagnostic only. State int8 safebounded9tokens,not general memory design. No model deployment/checkpoint saved.
