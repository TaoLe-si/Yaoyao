# Grouped AVX512 decision

Exact32full logits/states and advance states passed, generatedtrace same. AB/BA groupedAVX2 384.171/344.877 vs groupedAVX512368.181/374.753 tps. Aggregate363.465 vs371.438 (+2.2percent) with opposite pairwise directions. No convincing stable improvement. Keep groupedAVX2 preferred; no more width-only variants without new bottleneck evidence.

Absolute rates vary significantly under concurrent training; earlier~500tps resident observations are not comparable same-load benchmarks. Investigation continues with grouped-path phase profiling, not generic kernel sweeps.
