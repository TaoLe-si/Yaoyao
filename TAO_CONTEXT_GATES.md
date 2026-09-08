# Sparse current-token conditioned Wbi gates

Implemented logits=base_logits+sum(k=1..min(15,t)) q[x[t],k]/32 * Wbi[x[t-k],:]. Same full Wbi reused, no projection/quantization or duplicate vocabulary. q in{0,+/-1,+/-2,+/-4,+/-8}; each current token max2 nonzero distances, sum_abs(q)<=8.15KiB control table. Q1/head/Wbi/original reader frozen in first phase.

GPU train four4096-target batches offsets2000000,12000000,42000000,82000000. Support>=64 required. First bounded run searches all15 distances for top4 training token IDs1,3,6,5 (counts1833,1467,711,694); candidate selection never uses validation.8 accepted updates; train3.868917999->3.865222094. Token1 isUNK in production vocabulary. No claim of full vocabulary training.

|slice|zero gate baseline|trained gate CPU|
|---|---:|---:|
|0|4.162131982|4.162212676|
|1|4.026422611|4.023277908|
|2,4096|3.735809289|3.729963159|

Zero-gate GPU CE4.162132008 matches previous base GPU. Nonzero synthetic probes (+1/32 at distance1,-2/32 at15) compared CPU node.pop and GPU across1024x1024 entries exact max0. GPU compiled --fmad=false for this arithmetic gate. CPU loaded trained slice0 CE4.162212676 vs GPU4.162212703.

TCG1 tao_context_v1.tcg length15480:20byte header magic0x31474354/version1/V1024/D256/retained16;model/train/original-readerSHA256;15360 gate bytes q+8;CRC32. CPU loader validates header,length,CRC,model/reader binding,allowed values,sparsity,budget. Exclusive create/flush and byte roundtrip. Requires tao_alternating_step47002.bin + tao_coef_rts1_128.reader.

Sources train_tao_context_gates.cu and eval_tao_context_gates.cpp. Experiment entry retains legacy arguments rounds/start but bounded search currently hardcodes top4 token rows; these are not general resume controls. Parameters persist for CPU evaluation; training resume/expanded support next phase not yet implemented. Joint Wbi/head gradients not added: intentionally frozen first phase.

First evidence of useful nonzero full-Wbi contextual gates, small validation gain on2/3 slices, not broad capability/convergence claim. Baseline preserved.
