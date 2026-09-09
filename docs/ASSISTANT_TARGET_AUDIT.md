# Fixed BPE assistant targets

Train2015docs2564assistantturns1emptyturn398141supervised targets:395577body2564TURN_END. Endtargetfraction0.006439930;emptyfirsttargetfraction0.000390016. Validation23docs27turns0empty4906targets4879body27end fraction0.005503465. ExactBPEgrammarreader used; data unchanged, test split not inspected for tuning.

Step40 immediate greedy end cannot reasonably be explained as training examples predominantly empty or mosttargets beingend. A frequent special token can nevertheless be top1 in poorly learned conditional distributions while its absolute probability is low. No logitsdistribution measured here; no rootcause claim. Do not hidefailurebybanningend orchangingmask. Need continue fixed training and assessfuture savedmodel. CPUaudit only,no GPUcontention. Source audit_bpe_targets.cpp.
