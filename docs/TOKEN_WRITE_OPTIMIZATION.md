# Token selective write optimization and CPU budget

## Scope and reproducibility
Fixed512unique training sequences generatedseed58191;Net initseeds42,123,2026,7,999;batchorderseed12345. optimize_token_write.cpp compares originalSGD1200fullbatchlr.15 to Adam3000stepsbatch32lr.003,beta.9/.999,eps1e-8,L2.0001. SAME857FP64parameter architecture. Adam changes optimizer,batching,updates jointly;not isolated causal optimizer claim. NativeCPU training,not deployed/QAT. Weights saved local build/token_write_<optimizer>_<seed>.weights raw857doubles;diagnostic no production container.

Test query+token sequence exact dedup againsttrain and within test(noise0n128,others256). Noise0 rejects1258train overlaps in fixed draws;remaining unique finite set. Tests noise0,2,8,32,128. Samegrammar/IDs,not unseen-template/language. All5Adam seeds hard/softaccuracy100percent everytestedlength. TrainCE .01023..01209. At128noise softCE .5425..6639:leakage still real.
Same-data SGD seeds42/123 hard100percent alltestedlength but softfailslong;seed2026trainhard62.3percent,testhard~28..41percent fornoise>0. Thus initialization/optimization instability persists with fixed data,not solely previous dataset differences. Adam mitigates in5trials,not universal convergence theorem. More training data512vsprior256also changed from earlier round.

## Fault localization
inspect_token_gate on valid delimiter windows across4queries4objects8owners:SGD2026relevantcontent32/32 correct,but irrelevantSkip56/96;maxirrelevantp.999608. Adam123/2026relevantwrite32/32,irrelevantSkip96/96,content32/32. Adam42writesonly20/32atdelimiter yet hardanswerscorrect;some writes occur earlier;delimiter-only stats not globalgateaccuracy. No manual gate timing enforced.

## Hard implementation
token_write_hard.hpp validatesfiniteweights/inputranges;16tanh,gate dot then comparezero(no sigmoid),ifskip no content scores;write computes8logits,argmax(no softmax). Fixed previous2tokens plus ownerint;no sequence cache/allocation in step. No-write defaultunknown-1 intentionally differs uniform soft argmax ifneverwrite;tested predictions match reference. Hard paths overwrite memory,not reversible. FP64parameters not quantized.

test_token_write_hard five seeds:384sequences/seed spanningnoise0,1,3,16,128,512;1920/1920 reference hard agreement and correctanswers. Counterfactual perseed256irrelevantownerchanges and256relevantownerchanges allpass(1280each). New randomseed321991,not dedupaudited againsttrain forshortsamples. Counters vary onlyownernotgrammar;no languageclaim. Hardwrite rate aggregated.22..34percent due longnoise;not representative largeLLM writefrequency.

CPU AMD Ryzen9 7940H,8cores16threads. MSVC14.44 /O2 /arch:AVX2,no affinity/core isolation.9rounds x500fixed195tokenstreams,volatile checksum. Medianhard.118037,.122437,.124115,.128653,.129654us/token acrossseeds. Hot smallweights857*8=6856bytes. This is symboliclookup benchmark,not LLMdecode latency.

## Dense large-dimension surrogate
benchmark_write_gate_scale.cpp:FP32 denseconcat4D->H,tanh,gate dot;write addsH->Dcontent;ordinary scalaraccum loops compiledAVX2,not tunedGEMV/ternarykernel. Syntheticresidentweights andinput already available;no LMhead/state integration.7roundsx100,not pinned;true runtime may change under cache/bandwidth contention.
D256,H16skip9.799us write10.839us;H32skip20.211/write23.326
D1024,H16skip41.910/write48.423;H32skip82.033/write93.464
D4096,H16skip166.609/write180.336;H32skip333.583/write368.764
Costsroughly4DH+H+rho*DH MAC/token,weights5DH plusbiases. D4096H16 weights327680FP32=1.25MiB;H32=2.5MiB. FP32state+3inputvectors~4D*4=64KiBforD4096,excludinghidden and framework. If reusing existingstatebuffers persistentincrement differs. Ternarypacking could shrinkweights but unmeasuredspeed;do not project observedFP64small speed to it.

AtD4096H16,32separate layers naive linear extrapolation5.33..5.77ms/token;H32 10.67..11.80ms/token. ONEglobalgate.167..180ms or.334..369ms. These are arithmetic extrapolations,not measured32layerLLM. Ifbaseline20ms/token,32layerH16adds~27-29percent;baseline100ms adds~5-6percent. Differentbaseline needed foractualmodel,no universalpercentage. Layerweights40MiB(H16)or80MiB(H32),cache behavior differs.

Vocabulary:do NOT create new per-step V-class contenthead just to selectwrite. H->V vsH->D or existingfeatures;atV32768H16 extra524288weights,atV131072H16 2097152,scaleslinearlyV. ExistingLMoutputVxD already dominates many models,but gatecost not free. Recommend one/fewshared smallgates,existingcontext features,continuouscontentdim bounded,not8ownerclasses hardcodedforLLM.

## Assessment
This finite symbolic grammar subsystem now stableunder5testedinitializations with clear reference/gradchecks/hardsemantics and counters. NOT complete languagearchitecture:knownquery,fixed3tokengrammar,8owner taskmemory;no multiobject/posthocquery/unseentoken generalization,QAT,causal recurrentintegration or GPUtrainer. Soft leakage unresolved atarbitrarylength;deployhardonly ifvalidated. Next acceptance requires actualnewmodel context/input specification before claiming realLLM overhead. No legacycheckpoint/server modified,no push.
