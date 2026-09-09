# Fused greedy output head

StrictFP AVX2 unit tests pass: every scalar output bit on synthetic rows, serial/parallel, tails, tie handling, role exclusions, nonfinite rows including excluded roles. Frozenstep36032states/token bits exact; paired128trace/finalstates exact.

Pipeline full logits+scan417.272tps versus fused greedy431.361tps (+3.38percent); stagefull400.731. Small same-run gain. Greedy-only API omits full vector, not valid replacement for temperature/top-p consumers. Resident integration pending. CPU-only tests, GPUtraining unchanged. Unit mode was redundantly run before default (which already includes it); do not repeat.
