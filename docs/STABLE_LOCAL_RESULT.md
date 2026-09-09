# Stable local ordered accumulation: reject as sole latest-fact memory

Native diagnose_stable_local.cpp compares64int32state encodings:0cyclicpermutation+ordinaryadd;1sum current trit*shift7previoustrit;2sum current*shift7previous*shift19older. Fixedrandom16token64trit table;24tanh1760parameter floathead,1200fullbatchSGD.15L2.001. TaskfixedA verb object preposition B query,querysender/receiver.336train112heldoutcombination,noise0train;test suffixnoise0,1,3,16 beforequery. No rawlanguage.

Accuracy seed42/123/2026:
permutation noise0:100/100/100;noise1:9.8214/13.3929/12.5;noise3:8.9286/7.1429/13.3929;noise16:16.9643/8.9286/12.5percent
bigram noise0:100/100/100;noise1:48.2143/38.3929/49.1071;noise3:47.3214/44.6429/44.6429;noise16:22.3214/18.75/22.3214
trigram noise0:96.4286/97.3214/97.3214;noise1:46.4286/45.5357/40.1786;noise3:33.9286/42.8571/29.4643;noise16:16.9643/26.7857/12.5

Local encoding avoids globalrotation but suffixchangesquerylocalcontext and addsinterference;these effects not isolated. No robust lengthgeneralization. Same-datareadoutbudget,not trainedembedding/phi.

## Structural recency failure
CompareeventA+padding+eventB+padding+query versus reversedblocks,labels require latestownerBvsA. Sumlocal modes56/56statecollisions all3seeds;permutation0/56. Stronger independenttest_local_sum_limit.cpp checks EXACT histogram of slidingwindows widths1..16,56conflictingpairs each. Paddingwidth>=window resetslocalhistory. All896pairs equal histograms.
For any position-independent fixed finite-windowfunction phi, state=sum_t phi(window_t)=sum_w count(w)*phi(w). Equalwindowhistograms imply equalstate forANYparameters,dimension,nonlinearity. Blocksreset tosamecontext contribute additively;swappingblocks cannot encode whichlast. Samefinalquery/localcontext,so retaining lastwindowdoesnotfixthiscounterexample. Applies also mean pooling(equal length)and pointwise finalreadout;does NOT applypositiondependentphi,globalcontextphi,noncommutativeupdate,recencyweighting or externalmemory. Balancedpairedlatestlabels impose<=50percent regardlessreadout. Therefore no further training needed to reject THIS family as sole current-factmemory. Does not prove gate onlysolution.

## CPU update microbench
64dimensions only,1000tokens x200reps x5rounds,MSVC14.44 /O2 /archAVX2,hotrandomtable,volatile checksum,nohead. Medianacrossrounds byseed:permutation .028792/.023979/.036039us;bigram .043501/.037448/.041264;trigram .054393/.060573/.054711. AllO(D),naiveimplementations. Localnot fasterhere;not fullLLMtiming or optimizedkernels. Wideraccumulator int32 testedbounded1000tokens,no infiniteoverflowclaim.

Decision:retain localfeatures only as optional featureextractor,not standalone memory replacement. Latest-facttaskrequires mechanismbreaking blockexchange invariance:selectiveoverwrite,noncommutativerecurrence,explicitage etc. Existing learnedgate backup solves differentstructuredgrammar;no apples-to-applesperformanceclaim. Noattentionnecessityproof. Newmodel remainsunselected,noonlinechanges/traininglargecorpus.
