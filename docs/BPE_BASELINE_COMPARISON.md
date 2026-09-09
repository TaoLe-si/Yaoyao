# Fixed full-train-count baseline

Baseline counts only assistant supervisedtargets acrossall398141trainingtargets; add-one pseudocount each16384vocab item fixedbeforeevaluation. No conditionalstate, no learnedneuralweights, no optimizer. Validation23docs4906targets,14unseentarget occurrences. ValidationNLL8.811622545 versusuniform9.704060528. Step40modelCPUvalidation8.909300780 is0.097678235worse thanthisunconditionalcountbaseline.

Exposurecaveat: baseline sawalltrainingdataonce; modelstep40 hasnotseenfullepoch. Thus not an equalcompute/exposurecomparison or proofarchitecturecannotlearn. Still usefulminimumengineeringbaseline: currentnetworknotoutperformingthissimplefixedreference. No hyperparamsearch, no testevaluation, no changes to live training. User2.5notremotelyachieved.
