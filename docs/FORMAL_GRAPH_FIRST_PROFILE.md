# Round9 formal configuration CUDA graph

SortedGpuTrainer in dual_state_sorted_trainer.cuh preserves legacy prototype. New sorted projection version distinct;checkpoint version tracking pending.

profile_formal_graph.cu approved full configuration and initialization seed713:30185984trainableparams. GPUtrainer initialization .593s, free6490MiB. Additional projection .176s wall including host finitechecks/transfers, not pure kernel time. One token256 forward with target257 diagnostic CE9.701350 and fullbackward .024s. No optimizer update/no dataset training. Role target intentionally unmasked for graph profiling, not SFT protocol. FreeMiBafterone token6490 coarse allocator measurement, not zero activation cost.

One formal-shape token fits, not batch4 block256 capacity proof. One-pass timing not statistical throughput. Next bounded full-block resource audit and microbatch scheduler needed. Core correctness checked smallshape earlier;no redundant full finite differences. Goalactive. New profile_formal_graph.cu.
