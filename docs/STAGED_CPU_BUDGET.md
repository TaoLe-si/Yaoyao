# Round19 staged dense CPU surrogate

benchmark_staged_commit.cpp C++17 MSVC14.44 /O2 /archAVX2,7940H,FP32naivescalaraccumulators,hotprepared5Dinput,7rounds240tokens. Boundary4D->4tanh everytoken;gate5D->8tanh atcommit;content4D->8tanh->D onwrite. Headsreducedsumsratherthanlearnedscalarprojections,negligibleforlargeDbutnotexactLMkernel. Schedulesfixedbytokenindex,notmodelpredictions. Noembeddingfetch/L Mhead/KV/optimizer/stateintegration orternaryunpack.

mode0c0w0;mode1c1/3w1/12;mode2c1w1. D256medianus2.416667/4.973750/14.132917;D1024 9.970833/20.182917/57.655417;D4096 43.632083/82.952083/233.631667. Weights96D FP32=1.5MiBD4096. OpsD*(16+40c+40w)MAC. Inputvector5Dfloat=80KiBatD4096,output/stateextra/notfullymodeled.

D4096onceglobalmodule .0436/.0830/.2336ms/token. 32layers lineararithmetic extrapolation1.396/2.654/7.476ms/token,NOT32layermeasurement;48MiBweightschangescache. Ifbaseline20ms,~7/13.3/37.4percentextra;100ms~1.4/2.65/7.48percent. Actualnaturalcommit/write ratesunknown,donotimportsymbolicrate. Previousnaive4D16gate166.6usnotapples-to-apples:stagebranchdims/inputs differ. No universalimprovementclaim.

Decision:one/fewsharedcommitmodules preferabletocopyingeverylayer;reuseexistingfeatures;do notaddnewVclasscontenthead. Throughput1/(baseline+delta),notdirectrelativeus-to-token/s shortcut. fp32resultsnotpredictternaryspeed,quantizationaccuracyunverified. WholeLLMbudgetrequiresrealnewmodeldimensionsandbaseline.

Remaining scopedobjective evidence:mechanismsfoundunderauxboundarysupervision;answer-only/realtextstillunresolved. Finalroundshouldconsolidatewithout claiming universalroad found or markingunmetgoalcomplete. No genuineexternalblocker,justremainingresearch.
