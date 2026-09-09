# Shuffle exposure tests accepted

Real corpus four seeded helper permutations:all116steps655038positions398141supervised2015docs. Everydoc/reset/targetmask coverage and serialized remaining-work equality tested by helper suite. NoCPU neuraltraining.

Per-update supervisedrange varies1697..5844,611..5524,802..5896,1088..5679 vsfixed1250..5581. Thus shuffle does not inherently remove batchimbalance. Observed full-epoch occupancy same68.93percent for these4seeds, not guarantee allpermutations.

Running40step GPU pilot uses separately recorded std::shuffle permutation, not helper permutation, and finalDSB nonresumable. Do not conflate helper resume tests with GPUcheckpoint resume support. Parent compare recorded exposure atcompletion.
