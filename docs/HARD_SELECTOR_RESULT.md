# Dedicated hard selection inference

Implemented in tao_joint_selector.hpp:hard_forward(Input) retains strict full input validation;hard_lazy(original,logits,candidateCount,fetch,alpha,beta) validates scores/original/strengths and only fetches selected content. Skip performs zero fetches;Share/Add one. Does not compute exp,softmax,all-branch mixture or gradients. Tie policy remains first(Skip). No persistent state mutation. Lazy contract intentionally does not validate unselected content or temperature (unused);caller owns validity of score construction/candidate source. Selected content finite/shape and result overflow checked.

Tests:test_tao_hard_selector.cpp,1020random cases N0..16,D31,bitwise equality with reference hard outputs;Skip signedzero retention,zero fetch,one fetch and tie checks PASS. Existing640finite-difference selector tests PASS unchanged.

Microbenchmark N16,D256,9rotating rounds x2000calls,100warmup:reference computes soft+hard median18.344050us (min17.849650,max26.382650);strict hard11.415250us (10.161650..13.216550);lazy hard1.627350us (1.561500..2.968000). Lazy benefit includes avoiding full candidate finite scans,not merely eliminating exp. Inputs and scores already computed,values already resident;no token embedding,matching projection,LM head,network included. CPU not pinned;this is NOT end-to-end generation speed or guarantee.

No online integration,language training,loss change or checkpoint alteration. Soft training reference retained. Scores still must be computed before hard selection;zero content fetch does not imply zero cost to create scores. Build project script includes test. No commit/push.
