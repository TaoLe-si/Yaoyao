# UNK and EOS supervision audit

Performed after cleaned-source commit31e27805738348edf6360b51e079527a5a1f9b3b was pushed to origin/main. Existing corpora/checkpoints were read only.

|Corpus|Tokens|UNK|UNK percent|EOS|PAD|
|---|---:|---:|---:|---:|---:|
|Train|111816960|13245041|11.845288|0|0|
|Eval|4160|385|9.254808|0|0|
|Bound slice0|4160|447|10.745192|0|0|

The training corpus contains no positive next-token EOS examples. Suppression is not the only reason termination is absent; the data supplies no EOS supervision. Do not insert EOS after arbitrary64-token windows: actual story boundaries must be recovered from source records and validated.

Current Vocab::build preserves case but encode lowercases words. Both server and training IDs consistently reproduce this historical bug. First5Mbytes:1150410 tokens,133795 UNK(11.630201%). Missing lowercased forms include lily5602,i4780,once4296,timmy2146,tom1992. Consistent lowercase construction+encoding on exactly the same text reduces UNK to89590(7.787658%); this is a measured in-sample vocabulary coverage check, NOT validation or trained-model gain. Remaining OOV comes from limited1024 vocabulary and other tokenization choices.

Do not change existing vocabulary IDs or retokenize in place. Corrected datasets need new tokenizer/version manifests, story-level split, explicit EOS at real boundaries, and new training checkpoints. Current linked TCG/TDS artifacts remain bound to old model/data.

Native implementation audit_token_supervision.cpp; full count log token_supervision_audit.log(local ignored). Next: separate controlled retrieval training task with heldout contexts/keys and fixed two-distance budget, before horizon expansion or changing the live model.
