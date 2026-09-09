# Local sparse association and ternary action foundation

Isolated implementation tao_local_association.hpp;not trained or deployed.

Features:own token embedding concatenated with mean of preceding local embeddings. Strictly causal;mean loses order and is only a small prototype context feature. Shared ternary projection and cosine matching choose one history candidate within horizon;ties choose nearest;zero-norm matches invalid. No separate QKV or persistent writeback. Candidate context may precede matching horizon but stays inside supplied prefix.

Action semantics:-1Skip returns original;0Share returns(1-alpha)*original+alpha*value;+1Add returns original+beta*value. Defaultalpha=beta=.25. Not multiplication by action code. No candidate forcesSkip. Finite Skip preserves bits including signedzero;disabled path bypasses candidate/scoring. Validate strengths[0,1],used vector shape,NaN/Inf and overflow. Device arithmetic assumes validated inputs.

Candidate content is selected token embedding;original reading must be dimensionally AND semantically aligned by caller. Shape checks cannot establish semantic compatibility. Not wired directly into legacy model.

Three-output ternary action_scores head takes current||selected content;orderSkip,Share,Add. Forward currently accepts caller-supplied scores. Tests manually initialize matrices;no learned policy or optimizer. Hard argmax gradient,soft training versus hard inference,language coreference tests and live timing remain future work.

Actual verification:CPU test_tao_local_association PASS for three numeric branches,bitwise bypass,no candidate,invalid values/shape/action/overflow,causal prefix invariance,horizon,context sensitivity,shared projection,three-logit head,no writeback. CUDA test_tao_action_cuda PASS30003elements(10001/action)exact vs independent CPU formulas. GPU candidate matching/training not tested.

Build CPU via build_project.bat. CUDA compile test_tao_action_cuda.cu with nvcc12.6 MSVC14.44 --fmad=false -O2 -std:c++17 -arch=sm_75 -allow-unsupported-compiler,output build/test_tao_action_cuda.exe.

No dataset processing,training,checkpoint change or online deployment. Previous uncommitted foundation work retained. No commit/push this turn.
