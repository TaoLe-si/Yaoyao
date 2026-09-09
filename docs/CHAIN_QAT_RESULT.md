# Fresh diagnostic split and ternary QAT result
Protocol fixed CHAIN_QAT_PROTOCOL.md before run. diagnose_chain_qat.cpp native CPU. Split1344train,448heldout same-length combinations,224heldout length3noise. Train noises0/1 independent oflayout;OODnoise3suffix. Same2layouts throughout;no unseen-template claim. Hash unorderedpair/object groups all variants;split differs but is not external untouched corpus.

seed42float train100 test100 OOD12.0536percent;QAT50.7440/48.8839/8.4821
seed123float100/99.7768/14.2857;QAT63.8393/60.9375/16.0714
seed2026float100/99.5536/8.0357;QAT58.9286/56.9196/10.2679

Float testCE .099344061/.171164994/.127401826;OOD4.789649815/4.666784700/4.630200298. QAT testCE1.077206539/.893165528/1.014160899;OOD3.764880432/3.318126453/3.441585264.

Conclusion:float readout repeats strong within-seen-length composition but fails unseen suffixlength;consistent with position-dependent extraction,not stable relational understanding. QAT fixedscale/fromscratch configuration substantially underfits even train;not proof ternary capacity impossible. Scales,initialization,STE optimization confounded;no tuning after test.

Effective matrix weights always ternary*fixedscale in QAT forward;biases float,activations/gradients/master weightsFP64.1760total parameters,1728quantizedmatrix/32floatbias. Master identity STE inside clip;clip each update. Initial master out-of-range coordinates gradient masked firststep. No derivative claim for quantizer. Prior MLP gradient tested,quantizer/packed parity not independently verified this round. No model checkpoint saved,no online changes,no speed claim.

State int8 storage safe for max9tokens only;wide coordinate not trit. Equal-storage alternatives remain unimplemented. Next priority:position/length robustness before costly full-scale data training;separate QAT optimization from architecture with float baseline.
