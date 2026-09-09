# Combined AVX512 + two-thread result

Split translation units compiled: AVX512 kernel, AVX2 guarded caller, strictFP noLTO. Hardware guard passed.32-step full logits/states and advance states bitwise exact; generated128position traces equal.

AB/BA paired trials: baseline275.101 and346.150 tps; candidate351.115 and338.089. Aggregate306.563 vs344.479 (~12.4percent). First pair gain, second pair slight loss; substantial baseline variance prevents stable gain claim or automatic default replacement. Previous resident425-465 observations are different conditions, not comparable absolute peaks. Retain AVX2 default; candidate opt-in only until profiling evidence supports adoption. No GPUtrainingchanges.
