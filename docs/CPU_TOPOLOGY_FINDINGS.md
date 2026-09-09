# Actual CPU topology

Read-only Windows probe:8physicalcores,16logicalprocessors,1group. SMT siblings0/1,2/3,...14/15. Each core1MiB L2; shared L3=16MiB. Decoder byteweightpayload~29.15MiB exceedsL3 capacity, but this alone does not prove measured DRAMbound. Probeprocess allows all16LPs; decoder eligibility must be checked separately.

Candidate affinity experiment only in separate decoderprocess: callerLP0,workerLP2 (different cores/sharedL3). No global/process affinity or priority changes, no GPUcontroller interactions. Pair not selected by live load, pinned not assumed faster.

See build/cpu_topology.log.
