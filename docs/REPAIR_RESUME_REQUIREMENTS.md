# Required before any long-running shuffled training

Current40step pilot is intentionally nonresumable DSBonly. Production originalcheckpoint remains untouched and STOPretained. No finalpilotDSB can restore Adam/cursor.

Future native resumable format must atomically bind:
- immutable source corpus/tokenizer/architecture identity
- masterweights,Adam first/secondmoments,optimizerstep,LRpolicy
- every slot recurrentstate and canonicaldocument+target
- epoch,seed,explicitorder and orderalgorithm identity
- parameter/snapshot checksum and sidecar checksum in final commitmanifest

Canonicaldoc IDs must map to permutationrank when using legacyphysical-order loader; provide reordered corpus view for targetlength validation. Keep standardproductionloader unchanged. Reject rank>=next,duplicate active docs,wronglength/outofrange,missing/tampered sidecar,partialtransaction.

Acceptance: interruptedGPUrun and uninterruptedrun same nextdata,logits,gradients,Adam update and checkpointbytes after same update; metadata CPUtests alone insufficient. No permission to silently reset mid-epoch productionstates.
