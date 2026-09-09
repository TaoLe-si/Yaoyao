# Paired order-policy pilot results

Both40GPUupdates finished, sourcecheckpoint/exe/corpushashes unchanged since captured duringfixedarm, initialvalidationequal6.648445815549. ProductionSTOPretained. NoCPUtraining.

|arm|finalNLL|last10mean|weightedtrain|supervised|positions|
|--|--|--|--|--|--|
|fixed|6.722661350279|6.706137220356|3.112071|140471|224383|
|shuffle|6.533165504341|6.555534265135|3.655683|130285|228696|

Shufflefinal improvementvsfixed .189495845938 andvsinitial .115280311208. Last10also better, meets predeclared furtherinvestigation criterion not productioncure. Differentprefixes721vs677docstouched overlap223, ~7.25percentfewer supervisedtargets shuffle. Single seed and40updates confoundedcontentselection; not full-epoch comparison.

One-shot same9prompt CPUquality:fixed8caps,shuffle3caps. Shuffle often answers only当然可以 or irrelevantmusic, failsarithmetic/capital/translation/recall. Fewerloops NOT usefulassistant achieved. Raw build/repair_quality_comparison.json.

Decision: reject claim trainingfullyfixed, retain shuffle as promisingorderpolicy; dataexpansion next priority given matchedfrozengap and tinycorpus. Do not resumelongproduction merely becausepilotNLLlower. FinalDSBs build/repair1200_fixed.dsb and repair1200_shuffle.dsb inferenceonly, not optimizerresume.
