# Cursor and batching repair evidence

CPU data-only exact epoch simulation (no neuraltraining) completed:116updates,655038inputpositions,398141supervisedtargets,2015docs. Matches sum(doclen-1) and dataset masks; no evidence cursor omits targets.

Supervisedtokens perupdate1250..5581 (4.46x), padded graph usefulpositionoccupancy68.93percent. This explains noisy perbatch trainingmeans in part but not fixedvalidationgap. Fixedcapacity padded positions expected by Graph; not claim measuredGPUutilization.

Shuffled order may change padding and updatesperepoch: compare samefullcorpus exposure and report optimizersteps, rather than implying equalsteps equals equaldata. Suggested experiment preserve partialcurrentepoch until trueexhaustion, compare following completeepochs under same LR with explicitgradientstepcountconfound.

Source audit_epoch_batch_balance.cpp; evidence build/epoch_batch_balance.log.
