# Expanded contextual Wbi gates: completed by parent

Parent personally built and ran train_tao_context_supported.exe after verifying delegated agent idle/no process/no executable/no log. No duplicate training. Frozen model47002, Wbi,Q1,prediction head and original reader unchanged. All40 token rows with total support>=64 searched, covering10635/16384 train positions.60 nonzero gates learned; all40 rows active. Constraints verified from saved binary: <=2 gates per row, sum absolute weights<=0.25.

|configuration|slice0|slice1|slice2 (4096)|
|---|---:|---:|---:|
|zero gates|4.162131982|4.026422611|3.735809289|
|4token gates|4.162212676|4.023277908|3.729963159|
|40token gates|4.159973585|4.014222257|3.714128839|

Train CE3.868917999->3.848710554. End GPU slice0=4.159973610 vs independently loaded CPU4.159973585. Nonzero probe CPU node recovery/GPU full-vocabulary additions max_abs0. TCG1 byte roundtrip exact; CPU loader verified CRC,binding,values,sparsity,budget during all3 evaluations.

Files tao_context_supported.tcg/.csv/.log. Load with eval_tao_context_gates.exe tao_alternating_step47002.bin CORPUS tao_coef_rts1_128.reader SEQUENCES tao_context_supported.tcg. First4 row traces reproduce earlier run. Saved gates by distance:1=37,2=18,3=2,4=3,5..15=0. Benefits currently short-range; greedy distance order and2-slot limit can favor early positions, so cannot infer distant history irrelevant. Gates select distances conditioned on current token, not arbitrary dynamic semantic query. UNK1 included; token IDs not reconstructed as strings. Same monitored validation slices, no Transformer benchmark or claim of universal capability.

Expansion improves all3 slices versus zero and4token controls. This supports proceeding to prediction-head adaptation as a separate experiment, not claiming full convergence. Per-distance support caveat remains (11rows have<64 observations at distance15).
