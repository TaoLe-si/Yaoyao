# Local datasets and reproducibility

Data stays local; Git publishes source only. Current train count111816960 tokens, count-prefixed little-endian int32 IDs in0..1023. Train SHA2564e72a0c33c3b7bb9abb0108bb1b2cb274bc79e6b384554810707e5f639b84f18. Bound slice0 SHA2567f727c1cc8ae65dc5306c1384287fd106c4d15a85278b2f7aafafa15ed1b722f. Paths under experiments/d256_nibble64_baseline retained to preserve artifact bindings and source compatibility.

The split is the first99% of the existing corpus rounded down to64-token boundary, tail evaluation. Existing monitored slices are not a new unseen final test. Raw TinyStories text retained as tinystories_train.txt for tokenizer verification; vocabulary construction uses first5000000 bytes as existing code specifies. No retokenization during cleanup. Checkpoints and gates remain local with strict SHA dependencies.

Historical dataset preparation provenance and old checkpoints are in external recoverable archive (local CLEANUP_RECEIPT.json). Current corpus regeneration is not yet provided as an independently verified pipeline; preserve exact local files rather than silently rebuilding different data. Subsequent UNK/EOS audit will establish tokenization/supervision limitations before any new training dataset is introduced.
