# CUDA Graph optimization status

## Verified
- Reusable fixed 4-slot x 256-step formal graph compiled and executed with async allocation, device-side input/state metadata, deferred backward synchronization, and shared trainer parameter nodes.
- Restored step60 complete update: graph construction 4.783102 s; update 12.911100 s for 4408 positions / 2344 supervised targets. Output SCP and DSB byte-identical to the previously validated device-batch update.
- Graph continuous step60->62 matched non-graph batch step62 byte-for-byte. Non-graph batch continuous/resume equivalence was verified separately; graph-specific resumed execution is pending.
- Formal 32-step capture: 61,728 nodes; changed input/mask replay max absolute difference 0.
- Full formal 256-step reusable graph has executed complete optimizer updates. Graph-specific restore with a rebuilt executable remains to be checked.

## Performance evidence
Previous same-condition observations: original 64.810441 s, optimized serial 38.542271 s, device batch 24.72--25.54 s, reusable graph 12.738--12.911 s after build. First build plus update was 17.694 s. These are bounded diagnostic runs, not randomized repeated benchmarks. Nsight Systems trace of the non-graph batch path recorded 216,576 kernel launches, GPU activity union 21.8% of first-to-last kernel span, and large launch/allocation/memset API overhead; no SM utilization claim.

## Safety
Formal STOP_TRAINING remains present. Original step60 checkpoint remains untouched. Production controller was not switched to graph path and no long-term training was resumed. Graph entry startup with STOP exited without update; preflight restore of step62 succeeded.

## Remaining before formal resume
1. Run Nsight on reusable graph and compare API/kernel launch counts against non-graph path.
2. Measure exact peak memory with profiler, not cudaMemGetInfo samples.
3. Integrate graph path into controlled runner with unique checkpoint prefixes and explicit STOP/sidecar semantics.
4. Run one authorized 200-update continuation only after these acceptance gates and verify validation NLL plus readable AVX2 dialogue.
