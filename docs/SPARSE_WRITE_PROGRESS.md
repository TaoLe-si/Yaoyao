# Sparse affine write exploration (active, not final architecture)
Same857parameter tokenwindowmodel/data/order/Adam3000batch32 as optimize_token_write. Changed sigmoid gate to clip(.5+.25z,0,1),derivative.25interior0outside. Piecewise smooth,dead saturated gates possible;not a universal convergence cure. Transitionm=(1-p)m+p*v affine,noncommutative generically;p0identity,p1overwrite noninvertible. Not disguisinggateasnewarchitecture.

5seeds42/123/2026/7/999alltrainhard100percent,dedupnoise0/2/8/32/128soft/hard100percent. At128noise softCE .001853059/.002569468/.002104042/.003918248/.004403313 vs sigmoid.54..66. Exactzerosallowed reduceleakage;not allirrelevantstepsguaranteedzero,nor arbitrarylengthproof.

Gradient2571coordinates(857each3randominitializations)max3.7945001221e-10. Hardtest5seeds1920/1920reference parity/correct throughnoise512;1280irrelevantcounterfactual and1280requiredchange allcorrect. Samehardimplementation .111..119us/token medians,hot smallFP64table,not significant speedgainclaim.

Untrained grammar test fixedsame semantic events:original[object,owner,delimiter]100percentall;reversed[owner,object,delimiter]21.48/21.875/25.39/32.42/28.52percent;insertdelimiter[object,delimiter,owner,delimiter]12.11/2.73/26.56/21.48/6.64. No naturalgrammar claim;external declaration that unseen syntax is equivalent is extra test assumption. Still demonstrates strong template dependence.

Active nextwork:train mixedgrammar under fixedbudget,keep heldout query/owner combinations and unseen delimiter placements,possibly expandlocalwindow only if information insufficient. Distinguish grammar coverage from memory retention. Do not announce generalLLM route complete. Weightsbuild/sparse_write_adam_SEED.weights localdiagnostic only. Sources diagnose_sparse_write.cpp,optimize_sparse_write.cpp,test_sparse_write_gradient.cpp,test_sparse_write_hard.cpp,eval_sparse_structure.cpp.
