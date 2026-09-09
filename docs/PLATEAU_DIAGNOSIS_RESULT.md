# Plateau diagnosis and quality test

Training stopped safely1204, SCP/DSB retained, STOPfile retained, controllerlock absent. No automaticresume.

## Verified
- Step1200 identical9prompt qualitysuite:8cap64,1EOT; basic tasks failed, repeated math/phrase outputs.
- Full1101-1200 weightedtrainingNLL3.9120601353,minimum minibatch2.5565598. Not overallnear2. Validation1200=6.648445815549 vs800=6.648595694 essentiallyflat;1204=6.704048865597.
- Dataset2015docs657053tokens398141supervised; repeats sameorder116steps/epoch. Recorded restarts116,232,...1160.
- No full-document duplicate/trainval overlap, but decodedanswers66 repeatedgroups368 occurrences acrossbothsplits;2crosssplitgroups shortanswers. Identical shortanswers not necessarily leakage. Validation27answers median145targettokens,p90=260,max918. Corpus multilingual essays/math/shortanswer mixture, not evidence of broad assistantpretraining.
- Effectiveweight800->1200 supportchanged13.14percent,relativeL2=.4931;1200->1204 supportchanged.62335percent(187919elements),relativeL2=.1041,common-nonzero signflips0,meanrelative rowscalechange.004837. Quantizedmodel NOT frozen. Abrupt support changes plausible noise source but causality unproven.
- Readonly codeaudit found no confirmed mask/normalization/staleweight error. Correctshift,assistanttargets,docreset,sharedinplaceeffective pointers; deployed build --default-stream per-thread recorded. ExistingCPU/GPUparity~1e-8.

## Prioritized interpretation
1.Small repeated corpus and weakgeneralization strongest concern: improving minibatchfit not matchedfixed evaluation. Need samecheckpoint frozen trainholdout stratifiedloss to establish gap cleanly.
2.Fixedorder periodicity may drive local drift; verify per-epoch matchedpositions, then separate seededshuffle experiment, not immediateproductionchange.
3.STE+supportthresholddiscontinuity plausible contributing cause; observedmovementrulesoutnoeffectiveupdates, not proof QATsolecause.
4.TBPTT256 limitscreditassignment; statecontinues acrosschunks/updates with oldweightstate. Standardtradeoff, not a demonstratedbug.

## Recommended next experiments, NOT executed
Frozencheckpoint matched train/validation evaluation with perdoc/content/EOT loss; then bounded shuffled-vs-originalGPUtraining fork with samebudget. Ifneeded isolatefloat-vsQAT and stateburnin variants onefactoratatime. Preserve official validationdata and pastresults. No architecture/LR/data changes yet.

Evidence: build/quality_step1200_results.json,build/plateau_train1101_1200.json,build/decoded_corpus_audit.log,build/checkpoint_drift_audit.log. Diagnostic sources diagnose_compact_checkpoint_drift.cpp and diagnose_decoded_corpus.cpp compiled/run successfully CPUinferenceonly.
