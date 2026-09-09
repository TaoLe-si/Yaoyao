# Byte ternary CPU decoder candidate

Same preserved formal step360 bundle, explicit AVX2 single-thread CPU, GPU training concurrent. 16 recurrent greedy positions: full logits and both states max absolute difference0, argmax identical. Forced128-position greedy benchmark (does not stop EOS; not user dialogue speed): expandedFP32 269.185 positions/s, byte ternary342.333 positions/s,1.272x single observation. Excludes model load and initial packing; includes full forward and argmax. Does not establish 4k tokens/s.

Candidate preserves each row scale multiplication before activation multiply and existing lane accumulation order. Rejects non-exact-ternary rowweights. Current prototype retains expanded source weights in addition to byte representation; memory footprint is not yet reduced, only hot matrix reads. Not deployed to production dialogue executable.

Historical audit in progress: GitHub v0.9.8 README claims120+tokens/s not4000; later v21 commit1b7d5c0747637dd53380fc25c4f7c58f67ab0a90 claims5609 tokens/s, comparability not yet established.
