# Round28 uniform training budget control

Two architectures, seeds7/123,all4partitions. Increase3000to6000Adamsteps fromsameinitialization/RNG,notloadweightswithoutoptimizer. Onlinefreshsamples meanbothcomputeanddataexposuredouble;notisolatedoptimizersteps. LR.003,L2.0001,batch32init2 unchanged. Saveunique long checkpoints,notreplace originals.

All6heldoutconditionshard100cells: shared5/8->6/8;separated1/8->1/8. Shared123shift1 repaired;shared7shift0still73..100%;shared123shift3still73.83..89.84%,samehardnumbersas3000. Sharedtrain100all.
Separated7trainhardshift0..3:1,.986328,.716797,.927734. 123:1,.990234,.998047,.998047. Someimprovementbutnooverallrobustness. Separated7shift2trainCE.6868andheldoutsoft19..94,hard52..77:cannotassumejustneedsabitmoretraining. No proofintrinsiccapacityimpossibility;optimizationclippingcanplateau.

Decision:stopblindstepdoublinganddo notadoptseparatedmodel. Sharedresidualcandidate improvesnotcomplete. Lasttwofailuresneedidentifiability/representationoroptimization intervention onallpartitionprotocol;preserveheldouttestasdevelopmentnotfinalbenchmark. Candidategatepreactivationmargintrace/soft-hardconsistencylearning couldtest,notyetimplemented. Goalactive. Newoptimize_expand_long.cpp,optimize_factor_expand_long.cpp. Originalweightsretained.
