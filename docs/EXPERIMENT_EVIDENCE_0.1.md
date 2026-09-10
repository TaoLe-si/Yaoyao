# Fresh no-FFN learning: selected bounded milestone

All five training runs initialized from scratch; no historical weights, Adam or cursor loaded. Architecture two layers d64 s16 m64 vocab261,69829 active parameters, ternary projected forward/STE/RMSNorm/AdamW, no FFN forward or retained optimizer tensors. Constructor temporarily allocates legacy FFN before removal; model schema remains experimental, not production serialized format. CUDA forward/backward/updates; CPU inference only.

Task eight values,two people, answer-only token loss,64 ordered value pairs. Partition by (b-a+8)%8:0..5 train(96 queries),6 dev(16),7 sealed test(16). All transformations preserve partition. Train RNG900 independent from augmentation RNG1200, initialization seeds2026,2027. Deterministic per-tensor initialization. Dev explored clean600,mixed600,curriculum600 seed2026; clean/curriculum repeated seed2027. Fixed lr .003,4 examples/update,clip norm1,300? NO:600 updates each. Clean-first curriculum100 clean updates then500 uniformly mixed among clean,single40,four40/41,repeat first fact,swap order,third person12. Sixth compound transformation withheld from training:swap+third+noise+repeat. Candidate fixed before opening test.

Final sealed test counts pooled across2 seeds,32 questions per column (same16 cases across seeds, not32 independent data points):
|training|clean|single|four|repeat|swap|third|compound|
|clean600|32|29|13|16|4|13|15|
|curriculum600|31|31|31|28|26|31|21|

Aggregate descriptive accuracy122/224=54.4643% vs199/224=88.8393%. Compound15/32=46.875% vs21/32=65.625%; swap12.5% vs81.25%. Slight clean decline100% to96.875%. Not proof of natural-language learning; tiny symbolic dataset and2 seeds only. Transform tests correlated. Third person item=(a+b)%8 remains a shortcut risk. Further optimization must use a NEW sealed test set because current test was opened.

CPU/CUDA parity on each of4 final trained artifacts:24/24 greedy tokens agree,maximum absolute logit error<=2.8611e-6. Checkpoint loader validates byte length; raw ordered master/moment/variance arrays each100 steps lack embedded metadata/RNG and are not standalone resume-ready production checkpoints. Reproduce fresh: build/fresh_noffn.exe 0 SEED 3 600. Test: build/fresh_noffn_test.exe build/fresh_noffn_600_aug_3_seed_SEED_step_600.bin. Production training remains suspended and untouched.

Selected milestone: retain no-FFN + clean-first varied-context training as a promising validated experimental baseline; do not claim fluent dialogue or production readiness.
