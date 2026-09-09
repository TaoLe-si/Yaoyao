# GPU utilization audit during live training

Round50: nvidia-smi query utilization/memory/power/clocks/temperature failed with Failed to initialize NVML: Unknown Error. No utilization percentage, power or thermal conclusion available. Do not report GPU saturation from token throughput.

Live log confirmed step43 positions6778 supervised4944 preupdateNLL8.9315842,104.120223sec65.098positions/sec. PID19232 established by previous execution; no additional trainer launched. Latest durable checkpoint40 (pending next50).

Inspected code remaining costs:
- Tape.linear forward per-token matvec and backward dy outer product,4slots serial. Not true batched GEMM.
- GradNode still cudaMalloc/cudaFree for value/grad on every operation despite improved device zeroing. Synchronous allocations/deallocation and small kernel count remain.
- RMS forward/backward serial reductions.
- SortedGpuTrainer.update transfers entire gradients toCPU for finite/globalnorm checks; project transfers all masterweights toCPU for finite checks and scales toCPU. Each gradient/masterpass ~120.7MB decimal (formal30185984params*4), ~241.5MB combined perupdate plus scales. This is optimizer-step overhead, not per-token; do not assume dominates without profiling.
- detach uses D2D copies and clears graph eachblock; no state D2H roundtrip in this implementation.

Next optimizations should isolate measured CUDA allocation/matvec time before invasive batchedbackprop. Do not modify or rebuild running binary. Continue existing200update run while prepare future changes. No extra GPU diagnostic workload started. CPU eval at saved milestones; not every step.
