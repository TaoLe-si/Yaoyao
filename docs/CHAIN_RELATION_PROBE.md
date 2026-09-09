# No-attention reversible chain relation probe

Native C++ test_chain_relation_probe.cpp. Fixed random16token x64trit code table;sum chain versus reversible32+32coupling. Per-coordinate a+=e_a+b*e_b;then b+=e_b+a_rotated*e_a mod3. Reverse second then first,known token. No attention,hash,token-history readout or old checkpoint. Final64trit state ->trained FP64 linear8class head. Chain/embedding fixed,NOT trained.

Task symbolic sequence[A,verb,object,preposition,B,query] ->B;8people,4objects,A!=B.224unique examples. Test if(min(A,B)+max(A,B)+object)%4==0,56test/168train. Swap partners same split,balanced conflicting labels for sum chain. No exact sequence overlap;shared names/template;no linguistic parser/pronoun. Shortcut caveat:target always at fixed penultimate position,so success would not establish semantic coreference.

Head zero init,1200 fullbatch SGDsteps,lr.15,L2.001,seeds42/123/2026codebooks. Training only train split.

seed42 sum train50% test19.6429% CEtest4.360556565;coupling train99.4048% test10.7143% CEtest5.834958216
seed123 sum train50% test8.9286% CEtest4.269814812;coupling train100% test10.7143% CEtest6.321485577
seed2026 sum train50% test14.2857% CEtest4.252629408;coupling train98.2143% test7.1429% CEtest6.086486945

All224swap comparisons per seed:sum224collisions,coupling0. All sequence inverse checks PASS. This checks each generated sequence from zero,not exhaustive arbitrary-state bijectivity (algebraic proof separate). Chance8class12.5%;test small56.

Conclusion:coupling distinguishes swapped histories and supports train fit but does NOT generalize on this split with fixed random features+linear head. Not proof attention necessary,not proof learned coupling impossible,not pronoun ability. Need learn chain or structured positional baseline,heldout templates,disambiguation and causal controls before claiming relations. No large data/training GPU run,old model unaffected,no checkpoint output. Executable build/test_chain_relation_probe.exe,MSVC14.44 /O2 /std:c++17.
