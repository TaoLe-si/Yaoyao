# Matched frozen evaluation result

Checkpoint1200 same effectiveweights, freshstate perdocument, identicalGPUforward/teacherforcing/mask. Prespecified lengthstratified23train sample5145targets NLL3.386032444914; full23validation4906targets NLL6.648445815549; gap3.262413370635nats. Effectivebytes unchanged, no updates/writes. Officialvalidationexactlyreproduced.

This demonstrates substantial gap on sampledtrainingdocuments even after removing changingweights/preupdate/stalestate differences. Sample is not unbiased fulltrainingNLL. Many longtrainingdocs NLL2.9-4.0 vs validation longdocs6.5-6.9; shortanswerweight alone not sufficientexplanation. Does not uniquely separate memorization/data distribution/coverage/architecture.

Next controlledshuffleexperiment tests order only, cannot add missingknowledge; do not expect shufflealone guarantee usable assistant. Preserve originalvalidation. Manifest build/frozen1200_train_validation.manifest.tsv; allperdoc results build/frozen1200_train_validation.results.tsv.
