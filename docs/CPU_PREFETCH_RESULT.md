# Software prefetch candidate

Same frozen step360 checkpoint;32step full recurrent state/logit exactness passed. Two alternating128position trials baseline348.243/350.950 vs prefetch277.438/285.130 tps. 256byte lookahead at64byte intervals slows this workload~19-20percent. Reject this candidate; no broad claim all prefetch strategies fail. Existing byte single-row and two-thread kernels unchanged.

AVX512+two-thread combined implementation remains independent background work. No further prefetch sweep without profiling evidence.
