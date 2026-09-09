# Round21 normalized squared write control

Frozen five init2 models, same data and logit shift protocol as DECAY_CALIBRATION_PROGRESS. New mode5 uses rate=p²,retain=1-rate. No retraining, no new parameters. Modes0/4 controls reproduced.

Results original/required-update out of320:
- bias-2: baseline320/316, rejection300/291, squared176/193.
- bias-1 all320/320.
- bias0 baseline319/317, rejection and squared320/320.
- bias+1 baseline35/36, rejection320/320, squared52/34.
- bias+2 baseline39/34, rejection162/164, squared37/30.
All state masses1. Normalization solves mass drift, not semantic loss. This is repeated diagnostic data used for controlled intervention, not new independent final set. Logit offset synthetic, not actual language/quantization distribution.

Analytic retention: scalar update (1-a)m+a*u retains coefficient product(1-a_t) of earlier memory. Constant nuisance a=p²>0 leads (1-p²)^n tending0. Not an information-theoretic proof against every vector representation: concerns this specific same-slot convex update and distinguishable old component. Exact rejection preserves coefficient1 only if gate keeps rejecting. Wrong hard accepts still destroy memory.

One ideal old onehot and one distinct new onehot: squared update new probability p²,old1-p². New wins only p>sqrt(.5), versusp>.5 baseline. Thus squaring also makes necessary replacement harder. Real models may write multiple times within event, so this is explanatory idealized bound,not direct per-event threshold guarantee.

Numerical recurrence/pow agreement tested in test_decay_retention_bound.cpp forp .01/.05/.1 andn128/1024/10000. Decision:do not adopt p² as simple cure;keep exact-zero preservation as candidate but solve incorrect relevance/commit detection rather than only rescale rates. Next train-time gate-margin or boundary-based comparison needs independent protocol and no cherry-picking. Goalactive. Files test_coupled_decay.cpp,test_decay_retention_bound.cpp. No production changes.
