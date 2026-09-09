# Four-token context and online data diversity control

Round2:diagnose_sparse_write4.cpp extends window3->4,hidden16unchanged,input44->57,total857->1065parameters(+208). Samegate clipping,loss andBPTT. test_sparse_write4_gradient all1065weightsx3initializations3195checks,max3.16884685159e-10. No kink derivative claim.

optimize_multiformat_write4 samefixed512data/order/Adam3000batch32 as previousmultiformattrainer. Trainhard99.8047/100/100 vs3token83.4/67.6/68.6. OODnoise32hard mixed30.08/22.27/82.81percent;reverse66.80/85.94/100;separator57.42/82.03/100. Context helpsfit,not stableheldout generalization. Extra208params confoundscontextvsparametercount;no equalparamcontrol.

optimize_stream_write4 identical4tokenmodel and3000x32updates but freshgeneratedtrainingexamples eachbatch,mixed3syntax/noise0..2,fixeddataRNG58191 afterinitial512 draw. Records alltrainingquery/token strings for testrejection. Original512 trainhard report is diagnostic pool,not exact online training accuracy. All3seeds42/123/2026hard/soft100percent on3syntaxes xnoise2/32 x256test each. Noise2CE .00225..00534;noise32 .00224..0358. Eachtestexactdeduptrain+withincondition;no semantic group exclusion,seen grammar/IDs. Testednoise32outside traininglength. Noise0evaluation deliberately omitted because online drawscover finitepool;do not loop waiting for nonexistent unseenexamples.

Changingdata diversity helpsstrongly at sameupdate/sampleprocessing budget;not moretrainablearchitecture. Could reflect broadercontextcoverage/lessmemorization;not proof causal solecause. Floating QATnot yetretested. Nonlinearsparsegate still task-specificmemory8class,querybeforestream.

Next:twoadditionalinitseeds,128/512noise counterfactuals and hard4tokenparity/perf;unseen delimitergap beyond4window must be explicit separateboundary,not assumed supported. Need learnqueryfrominput/multientity test before callinggeneralLLM route. Goalactive. Newartifacts sources above,localbuild/stream_write4_adam_SEED.weights etc.
