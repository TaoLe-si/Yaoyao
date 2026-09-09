# Gate initialization controlled diagnostics round4

optimize_init_write4.cpp same1065parammodel,sameonline96kexamples/3000Adamupdates anddataRNG asbaseline. init1 allweights*.5;init2 ONLY16gateoutputweights*.1 andgatebias=-1;init3allweights*.5 plusgatebias=-1. Originalsigma.15. init2meanp~.25ratherthan.5,stillclipinterior,not exactzero. Changesgatevarianceandbiasjointly,not separatedcause.

Screenedseed7/42threeconfigs. init1seed7failsdiagnosticpool36.3%,longtest~13..50%hard;seed42hard100butsoftdegrades. init3hardalmostall100butsoftlongweak. init2bothallhardsoft100atnoise2/32. Expandedinit2to123/2026/999:all5hardsoft100everycondition. Diagnosticpool512notcompleteonlinetrainset. Oldtestsplit reusedforselection:development evidence,notpristinefinal. Originalseed7samebudgetnoise32hardmixed68.75,uniform6000budget96.875vsinit2at3000 100. Do not claimguaranteed convergence.

Fresh eval_init_write4 usesdataRNG518239+syntax,noise512eachside,128samples3formats5seeds. All1920hardanswerscorrectand1920referenceparity. Lengthensoutside traininglength0..2;trainoverlapimpossiblebylength;withinbatchdedupnotchecked. Hardtiming samecode median.124..153us/token,not pairedspeedimprovement,initialization addsNOdeployparams/ops. NoQAT,posthocquery,multientitymemory orrawlanguage.

Conclusions:gate-specific initialbalance promising;uniformsmall initialization is not universally beneficial. Spectral A recipes not tested:currentmemoryJacobian(1-p)I,pdoesnotdependonmemory. Sparsegateidentitybiasmustretaingradient;init2p~.25not saturating.

Next goals:counterfactual tests acrossallthreeformats plusheldout identitycomposition,independent initseeds not usedselection,then factorizedcontentbranch orquantization dependingfailure. Do not markglobalroutecomplete. Artifacts optimize_init_write4.cpp,eval_init_write4.cpp,build/init_write4_2_adam_SEED.weights.
