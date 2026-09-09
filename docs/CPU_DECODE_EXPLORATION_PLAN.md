# CPU decode exploration continuation

User authorizes continued optimization toward historical~4k t/s without claiming equivalence or changing model math. Frozen correctness checkpoint remains step360; ongoing GPU training stays untouched.

Current successes: byte ternary weights~29.15MiB, direct compact loading~0.15s, head-elided prefill21-30percent lower promptcompute latency, tail headelision preserves replies/token accounting over continuation/reset. Complete generation remains~300-390positions/s depending concurrent load.

Rejected speed variants: four-row AVX2,2bit shifts,LUT unpack,prebound scratch,signmask—all correct but not faster. Exact-orderAVX512 modest possible gain, not broadly deployed.

Next: bounded two-thread persistent rowparallelism with large-matrix threshold; test full logits/states and same-work generation plus monitor ongoing training before deployment. Avoid per-operator thread creation and busyspinning. No new quantization without explicit error/quality evaluation.
