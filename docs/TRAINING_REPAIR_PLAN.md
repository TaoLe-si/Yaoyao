# Supervised repair protocol

Production remains stopped at1204, STOPretained. User authorizes repair experiments, not silent overwrite.

1 Frozen diagnostic checkpoint1200: prespecified length-stratified23train documents and all23officialvalidation, same GPUforward/effectiveweights; log sample IDs, fullhashes and targetcounts. Sample result not fulltrainingNLL or unbiasedglobal estimate. No updates.
2 Deterministic epochshuffle helper must prove everydoc once, originaltargetcoverage, same-seed sameorder, resume trace equality. Currentactive document and state kept until trueexhaustion. Checkpoint identity binds epoch/order; no accidental reset atresume.
3 Current bounded pilot explicitly uses matched fresh-epoch+recurrent-state reset in BOTH arms from1200 weights/Adam,40updates each; it is NOT continuation of the saved mid-epoch cursor. OriginalSTOP stays. std::shuffle seed20260909, exactorder recorded. Same sourcecheckpoint,LR,optimizer state, oneGPU sequential; distinct outputprefix and controllerstate. No changing validation data to manufacture improvement. Fixedorderbaseline vs shuffle only. Evaluate by fullvalidation and recorded supervisedtoken exposure; same stepcount can entail differing tokenbudget so report both.
4 No longrun until evidence: finite state, checkpoint resume, heldout trend, loop quality. Shortcomparison insufficient for general architectureclaims. If shuffle fails, do not sweepLR blindly; inspect frozen gap and QATsupport instability first.
