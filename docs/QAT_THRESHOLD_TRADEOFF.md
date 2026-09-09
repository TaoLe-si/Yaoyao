# Round50 QAT uniform threshold intervention

Frozen12QATmodels/no retraining. eval_qat_threshold originalandlast-owner-counterfactual2304each,sameprior583921data/noise128. Threshold0 correct2083/2083 fullpass3/12 worst99/192;+.5 2150/2149 fullpass4 worst112; +1 2142/2139 fullpass6 worst112; +2 1970/1957 fullpass3 worst100. Noonepassesall. Morefullpassmodelsnotnecessarilybettertotoaccuracy. Test-selected+.5notindependentcalibration;notdefaultchange.

trace_qat_threshold.cpp uses+.5posthoc4608tracepool481273. Baselinefromround49errors557=19miss+538pollution;newerrors/counts recordedbelow. No referenceparityclaimatnonzerothreshold;codeassignsinterventionpredictionafterreferencecall. Metadataonlydiagnostics. Lowerpollutioncantradehighermisses,bothrequiredmetrics. No scale/weight change,notQATcure.

Nexttargettrainingdecisionrobustness ratherthan furtherthresholdsearch. Needgate-specificquantizationprecision orQATnoise/commitboundarydesignbudgetcontrol. Packedternarydeploymentstillnotimplemented. Goalactive. Files eval_qat_threshold.cpp,trace_qat_threshold.cpp.

Threshold+.5 trace errors=337, missed=54, wroteWrong=5, polluted=278. Differentpoolfrom2304table,do notmixdenominators.
