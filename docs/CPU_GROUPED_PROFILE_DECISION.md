# Grouped decoder phase findings

Bounded128positions instrumented rollout: step249.502ms. Disjoint percentages: s6.55,m26.35,read10.06,FFlinear26.75,head18.40,norm1.50,add/bias1.34,nonlinear6.40. Nestedwait39.72ms/15.92percent includes worker computation/imbalance, NOT pure overhead. Timer calibration floor~.42percent excludes actual perturbation.

Reject add/bias fusion as next priority: maximum benefit small. Matrix buckets dominate; investigate one exact-order software-pipelining candidate, preserving single accumulation chain. No claim 4k t/s feasible; arithmetic reduction dependency remains. See build/grouped_phase_profile.log.
