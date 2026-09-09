# Phase wall-time profile

Checkpoint60 firstsupervisedblock148positions, referencekernels, synchronizedphaseboundaries. Forward .941215s(~39.9%),loss scalartransfer .000068s,backward1.270935s(~53.9%),detach/teardown .145734s(~6.2%),total2.357961s. Previousnon-supervisedcontextblockalsoexecuted:forward1.695621s,backward0,teardown.319247s. Thiscontextisrequiredcursorwork, notduplicatetraining. CPUwalltimeincludeslaunch/allocation/synchronization; DOESNOTseparateGPUcompute fromhostsubmission. NoNsightclaim.

Conclusion: optimizingoptimizerorblocklossD2Halonecannotfixdominantforward/backward execution. Graphteardownnontrivialbutnotmajority. Needinsideforward/backwardinstrumentationorCUPTI/Nsighttrace, thenbatched/fusedexecution. Noformalupdatesorcheckpointchanges. profile_block_phases.cu reused diagnosticgradientdumppathreference_block_grad.bin; checkpointartifactspreserved. FutureprofilersshouldavoidunnecessarydumpIO.
