# Graph training resumed

User authorized continuing the long-term training goal after verified acceleration. Started graph_training_runner.mjs PID32592, log build/graph_controller_start.log. Initial phase evaluates preserved formal step60; subsequent graph training targets260 (+200optimizerupdates) with evaluations every10steps and original LR backoff. No CPU neural training. CPU fixed validation and explicit AVX2 dialogue remain quality checks.

Evidence: graph fullupdate12.9s vs historical batch25.5s and optimizedserial38.5s; firstbuild4.8s. Steps61 and62 exact SCP/DSB agreement with batch reference; rebuilt graph resume SCP step62 exact. Nsight captured eight GraphLaunch calls for eight accumulation rounds; default graph-level trace does not expose internal kernels in kernel summary, so absence there is not absence of GPU work. No claim of full SM utilization or exact peakmemory. Remaining optimization opportunities do not negate demonstrated end-to-end improvement.

Old STOP archived as build/STOP_TRAINING.archived_graph_acceptance; all formal models/checkpoints preserved. New graph prefix with overwrite rejection. Old controlled_run_state untouched. Current goal not complete: quality NLL<=2.5 and useful dialogues remain unresolved.
