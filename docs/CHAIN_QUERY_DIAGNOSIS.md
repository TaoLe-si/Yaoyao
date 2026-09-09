# Query-dependent relation diagnosis (partial next-stage investigation)

Two layouts:[A,12,object,13,B] and[B,13,object,12,A,15,other_object,15];append14(receiver query)or15(sender query). TargetsB/A respectively. All layout/query variants share group split.896total672train224test,three fixed seeds,64state,1200steps sameSGD. Distractors and sequence length6/9 are correlated with layout;not isolated length generalization. Query token15 also used as distractor marker;last-position distinction explicit. Symbolic templates,not natural language.

Unconditioned linear head testaccuracy seed42/123/2026:
sum23.2143/6.6964/16.0714percent
coupling15.1786/10.2679/13.8393
permutation_mod3 12.5/12.0536/14.7321
permutation_wide49.5536/51.3393/49.1071;CE~.708.

Diagnostic supplied layout+query selects one of4separate linear heads(4xparameters,NOT learned routing):wide100percent allseeds,testCE.008242561/.009186439/.008441038;mod3permutation16.0714/9.8214/13.3929;coupling10.7143/12.9464/11.6071;sum14.2857/12.0536/12.5.

Interpretation:wide alone does not solve conditional role selection;oracle-conditioned readout recovers accuracy. Ordinary linear accumulation+linear head adds query-specific bias but cannot implement general query-dependent content extraction. Conditional heads show represented positional content is recoverable given external template/query routing. This is upper-bound diagnostic with extra structural information/parameters,not architecture success or evidence all nonlinear heads fail. Learned small conditioning still untested.

Corrected swap diagnostic to swap persons inside SAME layout/query/distractor sequence;original expanded first run compared against previous fixed template,so its swap counts invalid and not used. Corrected conditioned executable:sum896/896collision,others0;inversePASS allgenerated sequences. Corrected unconditioned source retained but not rerun after diagnostic-only correction;train features/results unaffected by that correction.

Files diagnose_chain_query.cpp and diagnose_chain_query_conditioned.cpp. No deployed changes/training data changes. Equal-storage controls,learned nonlinear readout,independent distractor/length splits remain pending,not claimed complete. int8 wide storage safe ONLY boundedlength<=9,not long sequence implementation.
