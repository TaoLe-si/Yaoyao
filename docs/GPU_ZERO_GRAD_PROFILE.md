# Device-side gradient initialization

TAO_DEVICE_ZERO_GRAD opt-in in dual_state_autograd.cuh; new parallel driver enables it, legacy default unchanged. Replaces host zero vector allocation plus H2D copy with cudaMemsetAsync on default stream. Value buffers unchanged. Still cudaMalloc/cudaFree per node; not pooling/batching.

Same checkpoint33, same first supervised block256positions237targets: previous5.032430s, new4.146711s (61.736positions/sec); one observation each, ~17.6% less time/~21.4% higher throughput, not statistical/end-to-end guarantee. No optimizer update/checkpoint from profiling. Affected masked-loss gradient reference test rebuilt with flag; result in tool output. Next work should prioritize actual training continuation and resource telemetry rather than repeated profile loops. STOP_TRAINING remains until deliberate restart; latest durabletraining33.
