# Round24 residual hard-path parity and CPU cost

residual_write_hard.hpp templateE16/32 supports exactsharedlayout ofdeep16/expand,finiteweights/inputvalidation,4tokenhistory,initialmemory0matchesreferenceuniformargmax0 (notunknown semantic). No allocation perstep,gatezero threshold,no sigmoid/softmax,contentscoresonlywhenwrite. Residual layers mustcompute beforegate,so cannot skip theirworkonirrelevanttokens.

test_residual_hard.cpp compileDEXPANDorwithout. Five seeds each192sequences3knownformats/noise128heldoutpairs,newdata938241+syntax. Bothmodels960/960referencehardagreement. Expandanswers927/960 (seed7 159/192),deep16 905/960. Sameinputdatasets,notnewtraining,nodupwithinconditionaudit. No prefixcounterfactualoraclecomparisonhere,justfinalhardparity.

Ryzen7940H MSVC14.44 /O2 /archAVX2,7rounds200fixedstreams,volatilechecksum,FP64hotweights. Expand medianus/token .702444/.741788/.733152/.739879/.754035;deep16 .473710/.473187/.475098/.461907/.438298. Currentkernel scalarloops+tanh,nooptimizedSIMDmath/ternary. Priorplainhard4token~.12us fromdifferentroundnotpairedsamebenchmark;expansioncostmaterial,do notpresentasfree. NoformalLLMlatencyprojectionbasedontheseonehotnumbers.

Evidence:riseaccuracycomeswithCPUcost (~1.5x betweenlocalresidualvariantsinthisrun),largerwidthstillfailsseed7. Needgate/contenttrace andpossiblylow-rank/nonlinearalternative withequalcompute beforeadopt. Directplain32widthcomparisonaccuracyexistsbutitsCPUtimingnotthisround. Scopefinitegrammar/8classqueryknown/answer-onlytraining. Goalactive. Filesresidual_write_hard.hpp,test_residual_hard.cpp.
