# Decoder-side Tao Attention: version 2 verified result

## Contract
Native CUDA training; native CPU inference. Frozen step50002 checkpoint, original RTS1 reader and TCG Wbi gates. Current-token-conditioned state mixture: sum_k g(token,k)*a[k,d]*r[t-k,d]. Codes:0=identity1;1..4=+{1,2,4,8}/32;5..8=negative same. Only distances1..15 gated; distances0 and16 unchanged. Union of nonzero TCG and new state override distances <=2 per token. New table magnitude budget<=8. Support>=64 total occurrences (not per-distance support). Four4096-target training batches,40 eligible rows,31 accepted overrides.

Artifact tao_state_supported_v2_full.tds: TDS magic with version2,15696 bytes; CRC and hashes bind model,parent,reader,TCG,train,bound heldout,Q1,Wbi. Exact Q1/Wbi equality to47002 verified before transfer. Old version1 smoke/separate-budget artifacts and interrupted tao_state_supported_v2.csv are not final results.

## Results
Train CE3.843033299565 ->3.837546320551. Monitored1024-token heldout CE4.145156153272 ->4.134662417904.

Independent CPU evaluation,4096 targets each:

|Slice|Baseline|State gates|Delta|
|---|---:|---:|---:|
|0|4.031425622170|4.027196985932|-0.004228636239|
|1|4.018789724616|4.014476777886|-0.004312946730|
|2|3.715668649575|3.715127157137|-0.000541492438|

Small improvements on previously monitored slices, not evidence of broad language competence. No validation-based coordinate acceptance.

## Verification
CPU/GPU state probes exact max_abs0 including t0,t15,t16,t63. cuBLAS/reference logits max_abs4.57763672e-5. Actual CPU serving forward tested617 windows/631808 logits against reference and saved-prefix state oracle: max_abs3.43322754e-5; top1 agreement617/617. Float accumulation tolerance, not bitwise logit equivalence. Shared chain primitive remains a limitation of this parity test.

Fixed preexisting serving bugs: incorrect Wbi checkpoint offset (optimizer state was mistaken for weights), vocabulary tie ordering and lowercase encoding mismatch, unstable temperature softmax and missing nucleus renormalization. Regression checks1048576 Wbi entries,4096 training token IDs,100000 nucleus draws, low-temperature and greedy. Prior claims that incoherent generation was entirely model capacity were unsupported and withdrawn.

## Live API
http://127.0.0.1:11434; yaoyao_api.py uses tao_d256_api_state.exe and version2 artifact. Startup confirms step50002 state_gates31. API process PID24628 intentionally retained. Full200-token stream confirmed in openai_state_v2_stream.log. Internal average0.435ms/token,max0.592ms,cumulative87.074ms,2296.90token/s; HTTP wall106.672ms in this run. Internal cumulative includes callback/pipe intervals, not pure matmul timing; HTTP timing is separately measured.

Real output excerpt (not rewritten):

> , there was a because , was little girl named very was , little . who we to play outside the play of him that special , the decided . help wanted when see noticed better pond he the his of toy in .

Full original text is in openai_state_v2_stream.log. Grammar and coherence remain poor; CE gain does not establish meaningful story-quality improvement. EOS/UNK remain suppressed and a six-token repetition penalty remains enabled; this is decoding policy, not a fix for EOS modeling.

Primary sources: train_tao_state_gates.cu, tao_state_gates.hpp, eval_tao_state_gates.cpp, tao_d256_api.cpp, test_tao_server_checkpoint.cpp, test_tao_server_forward.cpp. Logs: tao_state_supported_v2_full.log, tao_state_v2_evaluation.log, api_state_v2.log, openai_state_v2_stream.log.
