# Learned conditional readout without oracle routing

Native diagnose_chain_mlp.cpp reuses previous symbolic dataset/split:896examples,672train224test;two seen layouts correlated with6/9length and distractors;receiver/sender query final token;all semantic IDs seen. Input ONLY final64coordinate chain state,no explicit query/layout routing,entity table or attention. Frozen random token embeddings and cyclic permutation chain. Mod3 after each update versus ordinary addition. Wide state divided bysqrt6,mod3scale1 (rough scale control,not equal variance).

Head64->24tanh->8,1760FP64parameters;randomGaussian.08 seed+999.1200fullbatch SGDsteps lr.15 L2.001;no test-based stopping or tuning in run. Both matrices/backprop implemented native C++. Finite differences8coordinates/model example0,epsilon1e-6,all6runs max1.85201530761e-10;not exhaustive gradient check.

seed42 mod3 train90.9226percent test12.5 CEtest3.274057592;wide train100 test100 CEtest.039908276
seed123 mod3 train90.625 test11.1607 CEtest3.614658730;wide train100 test100 CEtest.046761922
seed2026 mod3 train90.625 test14.2857 CEtest3.513045145;wide train100 test100 CEtest.040550662

Compared previous wide+linear~50percent,small nonlinear head learns conditional readout for heldout combinations WITHOUT supplied layout/query labels. Head1760params versus previous linear520 and oracle4heads2080. No equal-compute or equal-storage claim. Mod3+sameMLP still near chance;not proof larger/trained trit systems impossible. Wide alphabet larger than ternary,possible capacity advantage remains. No long-sequence safe implementation:int8state boundedabs<=9for this task only.

Scope:seen-template heldout combinations,not unseen names/templates,variable independent noise/length,real language/coreference. Prior evaluations already inspected same split;this is diagnostic benchmark,not pristine final validation. Candidate architecture evidence only,no from-scratch language model/QAT/GPU training.

Reproduce MSVC14.44 /EHsc /O2 /std:c++17 diagnose_chain_mlp.cpp /Fe:build/diagnose_chain_mlp.exe;run42,123,2026. No checkpoint or production changes,no push. Next independent task/split and storage controls before adoption.
