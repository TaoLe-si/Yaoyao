# Controlled diagnosis of chain representation bias

Native diagnose_chain_bias.cpp extends previous fixed6token symbolic task unchanged:168train56test,8classes,64features,linear65x8head,1200SGD .15 L2.001,seed42/123/2026. All named tokens observed;heldout combinations. No attention,trained embeddings or chain.

Test accuracy by seed42/123/2026:
sum mod3:19.6429/8.9286/14.2857percent
random coupling mod3:10.7143/10.7143/7.1429
cyclic permutation+mod3:8.9286/5.3571/14.2857
same permutation+ordinary integer addition:100/100/100
wide features divided bysqrt6:100/100/100
oracle target token embedding:100/100/100

All sum swapped pairs collide224/224;other chain paths0/224. All inverse tests PASS. Oracle collision/inverse printed fields refer to underlying wide chain BEFORE target embedding override;NOT oracle properties.

Ordinary accumulation stored in int8 only because sequence length6 bounds absolute coordinate<=6,so no overflow here. Not a safe general long-sequence int8 design. Wide state has larger state alphabet/capacity than ternary state;not equal-bit-budget comparison. No additional trainable parameters. Scaled variant divides bysqrt6,rough magnitude control only,not exact empirical variance matching.

Inference supported:missing order causes sum ambiguity;adding order alone does not solve this fixed-feature linear readout task;removing per-step mod3 in otherwise identical permutation update restores heldout-combination accuracy. Suggests modular wrapping destroys easy linear component separation here. Does NOT establish universal failure of ternary state or isolate geometry from additional information capacity. Oracle rules out trivial label/readout inability on supplied target code,not end-to-end task solution. Coupling negative result alone does not prove nonlinear readout cannot decode.

Critical limitation:fixed template,target always penultimate token;wide chain can learn position extraction.100percent is NOT semantic relationship/pronoun understanding;no heldout templates,long histories,nonlinear readout,trainable chain or language model tested. Cannot attribute historical LM convergence exclusively to this probe.

Recommended next:variable target positions and queries,interference/length controls;compare wider nonlinear head on mod3 versus wide linear plus equal-storage controls;then train chain. Keep ternary embeddings/weights while considering wider semantic accumulation distinct from reversible trit bookkeeping. No production modifications or model outputs.
