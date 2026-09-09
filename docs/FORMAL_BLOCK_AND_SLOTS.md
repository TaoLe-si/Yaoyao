# Round10 approved block resource and slot isolation

Formal8layer512/128/512/1024/16384model,sortedQATforward/backward256tokens diagnosticbytecycle alltargets(unmasked),nooptimizerupdate/noabilityresult. Forward3.575s backward2.423s,free6418/6344/6272/6198MiBafter64/128/192/256. Onepassclocknotstatisticaltrainingbenchmark. About292MiBadditionalactivation allocation observed. GPU fits single256block notparallelbatch4proof.

SequenceSlots serializes slots sharing parameter nodes/gradients; each separate s/m retained across chunks, resetnewdoc, no padding execution. Effective weights fixed until explicit update. This preserves microbatch gradient accumulation semantics but not parallelGPUthroughput;4slots*8accum may be slow ~minutes/step atcurrentlaunch-heavygraph. Need notbuild4simultaneousgraphs tofitmemory. Test2slotsinterleaved4chunks,eachlastpositionloss,comparedseparateCPUstatepathsmax1.699e-6andoneGPUupdate. Not full four-slot data scheduler yet;dataset cursor/slot checkpointstate remainspending. NoCPUtraining. Goalactive.

Filesprofile_formal_block.cu,dual_sequence_slots.cuh,test_sequence_slots.cu.
