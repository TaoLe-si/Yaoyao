# CPU row-parallel scaling

All four configs passed32step complete logits/state bitwise tests. Alternating forced128positions, GPU training concurrent.

|Threads|Threshold elements|Baseline mean tps|Candidate mean tps|Ratio|
|--|--|--|--|--|
|2|262144|352.155|422.416|1.200|
|2|1048576|331.111|360.322|1.088|
|4|262144|305.330|370.530|1.214|
|4|1048576|311.149|341.440|1.097|

Different paired baseline rates show changing host conditions. Fourthreads do not convincingly improve normalized gain overtwo; choose2threads lowerresource overhead. Lowthreshold parallelizes57/113matrices, high only vocab projection. Provisional2thread integration with directload +advanceprefill pending real resident verification. No claim 4k t/s or speedup from different conditions. No change to GPUtrainer.
