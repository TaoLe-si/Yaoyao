# Optimization gate before training resume

STOP_TRAINING stays present. Active objective optimization first, not automatic training restart.

Confirmed host traffic each update: full gradients read toCPU for norm; full masters and scales read for finite checks in project. Candidate gpu_health_reduce.cuh performsGPUdouble sum/finitecheck with12byte summarycopy instead of fullarrays. Notintegrated/notcompiled/notvalidated yet; summarycopy stillsynchronizes, mustnotclaimzeroCPUinteraction. Next moveclipfactorondevice andaggregateerrorcheck, thenGEMV/backward/RMS/batch execution work.

Acceptance: checkpointcompatibility, fullforward/backward/update numericalcomparisonincludingnonfinitefailure, identicalworkloadendtoendtimingwithtransfer/kernel/allocationbreakdown, peakmemoryheadroom; no utilizationpercentagealone. Keeporiginalpath. Do not resumeuntilusefulverifiedexecutionimprovementsandremainingbottlenecksassessed.
