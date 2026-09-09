# Architecture2: input scale and training horizon

User requests diagnosis, architecture detail revision, restart. Loss still improves, not proven plateau: step50CPU8.857912083,60CPU8.768404183. Old LR step60 .000126588769 vspeak.0003 andfloor100 inconsistent with233run extension; traininglessoneepoch. Measured effectiveembeddingRMS.039966,readnormgain~.25; residual token identity scale disparity motivatesinput-onlysqrt(D) hypothesis, notprovenrootcause.

Arch2: x0=sqrt(D)*embedding[token], outputmatrix unchanged tiedternary. Allotherstateequations/dimensions/masksunchanged. CPU/TrainGraph macrosTAO_INPUT_SCALE; bundleoperator dual-state-2-input-sqrt-d rejectsolddecoder. CudaResident unchanged and notarch2supported; use evaluate_arch2/decode_arch2 CPU artifacts, not legacyCUDA evaluator.

Oldrunstoppedandsaved63. Newtrain_yaoyao_arch2.cu explicitmaster-onlywarmstart63,resetAdam/state/datacursors/step0. New200updatehorizon,warmup4to3e-4cosine196to3e-5. Separateprefixyaoyao_arch2_step_. Notseamlessresume,notcleanfromscratch,andtwointerventionsconfoundcausalattribution; do not claiminputscalingaloneimproves. Oldweightsunchanged. MaskedendtoendCPUfinite-difference/CUDAgrad testpassedmax.00026645031. Actualpreflightwarmstart63 restoredstep0,nextdoc0,slots4passed.

Drivercurrentlywarmstartonly, requires63source; futurearch2checkpointresumehandlerneededbeforecallingitresumable. Newcheckpointsformatstoresallstatebutdriverdoesnotyetloadthem. Diagnosticloss/gates/quantizationsymbolchurnnotfullymeasured; no claimmaincausecertain. Goalvalidation<=2.5stillnotachieved.
