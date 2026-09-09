# Filtered long-run early observation (not complete)

Live PID 34952, 6/200 updates, no FAIL. Official 23-doc GPU val every step.

|step|val NLL|
|--|--:|
|0/1240 master|6.309478603453|
|1|6.294169974166|
|2|6.268375815343|
|3|6.255841269668|
|4|6.246624935095|
|5|6.244709069807|
|6|6.239853555905|

Monotone so far; 6 steps cannot establish oscillation cure or quality. Production STOP retained; 1200/1204 and expanded_repair_v1 not overwritten. After 200 steps only: analyze_filtered_repair.mjs then one-shot test_filtered_repair_quality.mjs.
