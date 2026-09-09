# Joint candidate/action trainability foundation

## Graph
N causal candidates ->1+2N logits:uniqueSkip,Share0,Add0,Share1,Add1,... . No per-candidate duplicateSkip. Shared ternary action head producesSkip(current,zero) and Share/Add(current,candidate),plus candidate match score. No pre-training argmax pruning;all supplied candidates participate.

Soft training output=sum softmax(logits/T)*branch,branch semantics unchanged. Hard inference=max logit branch;ties earliest(Skip first). Temperature finite positive;alpha/beta finite[0,1]. Zero candidates returns original. Strict shapes,finite inputs/results/gradients. Selection helper consumes candidates;does not itself build causal windows. Existing local feature builder remains prototype own token+preceding mean.

backward returns exact derivative of soft output with respect to logits,original,candidate values,alpha,beta for upstream gradient. This is NOT derivative of hard argmax and does not automatically backpropagate through matching cosine,ternary scoring weights or embeddings. Those integrations remain to implement. Temperature derivative not returned.

clipped_ste(master,scale,gradient) is explicit identity-inside-clip surrogate for weight quantization. True ternary quantizer derivative is zero almost everywhere;STE is biased surrogate,not justified by finite differences and not called by current forward/backward. No QAT optimizer/master checkpoint pipeline claimed.

## Actual tests
CPU640 finite-difference comparisons across40random cases cover logits,original,values,alpha,beta:maximum2.44484946177e-10. Empty candidates,ties,lowtemperature,nonfinite,shapes,ternary head output and STE mask checks PASS.
Toy temporary floating logit optimization400steps:loss8->.001038831,hard choice2(Add candidate0),soft3.954418614 versus hard4;target4. This trains only temporary scores,not head weights or language model;nonzero soft-hard gap remains.
CUDA tiny fixed two-candidate three-dimension oracle:soft outputs+logit gradients max_abs0 against CPU. Not GPU batch trainer/performance benchmark.

## Files and reproduction
tao_joint_selector.hpp;test_tao_joint_selector.cpp included build_project.bat;test_tao_joint_selector_cuda.cu compiled with MSVC14.44 CUDA12.6 --fmad=false -O2 -std=c++17 -arch=sm_75 -allow-unsupported-compiler. Executables build/.

## Next integration boundary
Need actual loss upstream from existing prediction head,content dimensional alignment,derivative through matching/score projection,QAT training decision,hard inference parity/quality and timing tests. Existing local module accepts externally supplied scores;this layer supplies computable ternary logits and output derivatives,not an end-to-end trained system. No Infinity processing,production checkpoint modification,online deployment or language training. No commit/push.
