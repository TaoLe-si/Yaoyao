# Grammar coverage tests, active architecture exploration

Read current sparse files and ran two additional nativeCPU protocols thisround. Same857params16hidden3tokenwindow,finalansweronly,Adam3000batch32lr.003512unique examples,threeinitseeds42/123/2026. Training changes grammar distribution,not weightsize. Test exactquery+tokens dedup againsttrain and withincondition,noise0n128,2/32n256. Fixedknown13tokenvocabulary,queryprovided. No independent semantic heldoutgroup split.

optimize_mixed_write:eachtrainingevent random[object,owner,delimiter]or[owner,object,delimiter]. All3seeds trainhard100%;mixed and reverse testhard100% atnoise0,2,32. Softseed123atnoise32mixed98.05%,reverse94.14%,others100. Unseen[object,delimiter,owner,delimiter] hardnoise2 12.11/12.5/12.5%,noise32 12.5/8.98/10.16%. Thus learning orderalternatives possible with no extra parameters,unseen separators not automatic.

optimize_multiformat_write includes separatorformat in training distribution(syntax draw persequence). Samebudget underfits:trainhard83.4/67.6/68.6%. Separatornoise2hard48.44/28.52/21.09%;noise32 36.33/25.39/23.83. Otherformatsdegrade too. Do not blame generalization alone;trainingfailure. No test-driven earlystop/bestseed selection.

Bothinternal andeventboundarydelimiteruse SAMEtoken12;3tokenwindows can confuse boundarycrossing localpatterns. Atterminalseparator in4tokenformat,object is outsidewindow,though available atpreviousownerstep. Not globalinformation-impossibilityproof:model maywriteearlier. Need window4control and distinguish optimization saturation/capacity from ambiguous localcues. Must not silently reinterpret delimiter as whitespace universally;grammar semantics external fixture.

Saved diagnosticweights build/mixed_write_adam_SEED.weights andbuild/multiformat_write_adam_SEED.weights. No productionchanges,goal remainsactive. Next examine4tokenwindow samehidden/sparsegate,gradient checks and pairedprotocol;if improves assessaddedcost and novel longerseparation.
