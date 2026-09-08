# Ordered Tao v2 diagnostics

Executed in order:1 same-binary ablation/timing;2 paired synthetic distance sensitivity/repeat retrieval;3 fixed-seed decoding policy sweep (then isolated EOS-only control). No checkpoint, trained gate artifact or live API setting changed. Native C++ driver includes actual ServerCore::forward_window and sampler.

## 1 Same-condition ablation
All modes use fixed50002 head, original reader and same forward binary.0=reader only (TCG zero, state identity);1=frozen Wbi gates,state identity;2=Wbi+trained state gates. This is inference ablation of a head trained with gates, NOT three separately retrained architectures. Default gates still execute shared gate lookup/mixing machinery, so timings measure active-value effects, not all machinery removed.

Three existing monitored slices,1024 targets each, sliding64-token windows at positions63..1086. This differs from previous4096-target reset-per-sequence evaluation, intentionally matches serving window semantics.

|Mode|Slice0 NLL|Slice1 NLL|Slice2 NLL|Top1/3072|Median us/token|
|---|---:|---:|---:|---:|---:|
|Reader only|4.132969108|4.100153683|3.788555935|648|325.617969|
|+Wbi gates|4.127537564|4.088055672|3.765417661|645|325.750391|
|+State gates|4.122232663|4.084024737|3.765936676|638|327.803516|

Timing:64 warmup windows/mode,9 rounds rotating order,512 identical windows/round, single-thread CPU C++ matvec. Includes allocation,window forward and checksum read; excludes sampling/network/tokenization/model startup. Not pinned hardware or statistically established speed difference. Baseline range323.79..334.19us;Wbi323.54..335.03;state324.34..355.19. Median state-minus-Wbi2.053us(~0.63%),ranges overlap; cannot claim previous0.18ms was gate overhead.

State gates lower NLL on two slices but raise slice2 by0.000519;top1 decreases7/3072. NLL improvement is not top1 improvement or story quality proof.

## 2 Distance diagnostic
128 paired contexts per distance, seed4242, same filler/current token and balanced key A/B. Swap only key at distance d from last input; score whether next-token logits favor the seen key over alternate. Frequent IDs3..127. d=2,4,8,15,16,32,64. This is an untrained synthetic repeat task with random filler and no explicit learned query instruction: primarily sensitivity/control diagnostic, NOT conclusive trained retrieval/general-language assessment. Pairs share filler/targets to reduce fixed unigram preference;128 pairs are small and no significance claim made.

V2 binary accuracy: d2=.503906,d4=.503906,d8=.500000,d15=.531250,d16=.503906,d32=.515625,d64=.500000. Near chance. Full-vocabulary top1<=.007812. Logits respond to key changes within64-window, but response does not establish correct recall. At d64 key omitted before forward (identical windows),max_logit_change exactly0: structural outside-window negative control, not a surprise learned result. A dedicated trained task with explicit query/rule and heldout keys would be required to assess learnability.

Distance counts:
- Wbi:1:37,2:18,3:2,4:3.
- New state gates:1:5,2:13,3:8,4:2,5:2,7:1;8..15:0.

Current search already permits1..15. Joint<=2 distance budget and greedy ascending distance search can favor early distances; these results do not identify the cause of missing long-distance gates.

## 3 Decoding strategies
2 prompts x3 seeds(42,123,2026) x7 policies=42 outputs,96-token cap each. Greedy repeats across seeds are deterministic duplicate controls, not independent samples. Same v2 weights,all prompts and raw token IDs retained in tao_diagnostic_stage3.log.

|Policy|T|top-p|penalty|avg distinct IDs/sample|adjacent repetitions(total576tokens)|special IDs|EOS stops/6|
|---|---:|---:|---:|---:|---:|---:|---:|
|Baseline|.9|.9|3|55.67|0|0|0|
|Lower T|.6|.9|3|41.50|0|0|0|
|Narrow p|.9|.7|3|48.17|2|0|0|
|No penalty|.9|.9|0|51.00|11|0|0|
|Allow EOS+UNK|.9|.9|3|43.67|16|107|0|
|Allow EOS only|.9|.9|3|55.67|0|0|0|
|Greedy|0|1|3|30.50|0|0|0|

Other policies mask EOS/UNK. Penalty applies last6 tokens with3*.65^back; EOS-inclusive policy keeps that penalty. PAD policy unchanged. Vocabulary diversity and adjacent repeat counts are NOT semantic quality metrics. No natural EOS in96 tokens does not prove EOS impossible. Inspecting sample excerpts still shows ungrammatical text; no blinded/human quality ranking conducted.

Unedited baseline excerpt(seed42):
> , there tree a ' on glad she had the a . , next have the you . ' smiled mom said that she started for the .

Unedited lower-T excerpt(same seed):
> , there ! a little . was wanted happy see thanked . for he little to who the to side . the took , her said .

## Decision
Do not expand context or change deployed decoding settings based on these tests. First audit tokenization/UNK coverage and EOS supervision, then build a small explicitly trained paired retrieval task with heldout templates/keys and distance-balanced support. Compare exhaustive allowed distance pairs or randomized search order against current ascending greedy search under same<=2 budget. Only extend horizon if demonstrated retrieval benefits or boundary-specific failures justify it. Current findings show small conditional NLL gains, near-chance untrained repeat retrieval, and decoding heuristics changing surface statistics without established coherent-story improvement.

Outputs: tao_ordered_diagnostics.cpp/.exe,tao_diagnostic_stage1.log,tao_diagnostic_stage2.log,tao_diagnostic_stage3.log,tao_diagnostic_stage3_summary.json.
