# Round9 smooth gate and L2 ablation

Samepairedaugmentation/factor8+8/init2/dataflow3000updates asround8. Replacedclip(.5+.25z) bynumericallystablesigmoid(z),exactderivativep(1-p). test_smooth_factor4_gradient3195coordsmax2.82709488632e-10 (printedlegacy sparse/kink label inaccurate;actualcode sigmoid). Samebias-1 givesp~.269vs.25,nearbutnotidenticalinitialgate. NoSTE,nohardtrainclaim.

Seeds2026/7/123:2026samehardaccuracyasclippedversion,diagnostic85.16%,heldout72..79. 7and123hard100allseen/unseenpairs. SoftCEgrowswithlength(~.13..61atnoise32). Henceclipdeadzone notsolecause,thoughcouldmatterelsewhere.

Single-factorL2ablation:smoothsameprotocolremove.0001*wterm.2026nowhard/soft100all,123also100;7degradesheldoutmixed91..96/reverse92..93,seenmixed96..98. ThusuniformL2affectsoptimizationpath,neitherL2norzeroL2universallydominates in3trials. No permanentconfigadoption,not complete5seedgrid. Testsreuseddiagnostically,noindependent finalclaim.

Files diagnose_smooth_factor4.cpp,optimize_smooth_factor4.cpp,test_smooth_factor4_gradient.cpp,optimize_smooth_nol2_factor4.cpp. Savedlocalweightsbuild/smooth*_factor4_SEED.weights.

Assessment:stopblindscalarhyperparametersearchon8+8. Need robustarchitecturalortrainingcriterion plus independentvalidation,not selectperseedconfig. Potentialnext asymmetriccapacity16gate8content vsfullsharedandfactor32,or semanticinvarianceupperboundexplicitlylabelednotgeneraltext. Originalshared16init2strongonseencompositions;factorizationnotyetprovedbetter. Nochangeformalmodel. Goalactive.
