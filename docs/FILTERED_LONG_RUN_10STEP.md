# Filtered long-run 10/200 (not complete)

PID 34952 alive, lr=.0001, official 23-doc GPU val every update, no FAIL, production STOP retained.

|step|val NLL|
|--|--:|
|0|6.309478603453|
|1|6.294169974166|
|2|6.268375815343|
|3|6.255841269668|
|4|6.246624935095|
|5|6.244709069807|
|6|6.239853555905|
|7|6.235135830532|
|8|6.234037061351|
|9|6.228944073774|
|10|6.225861583739|

Last10 mean 6.24736. Supervised so far 39904 / 3887672 (~1.0% of filtered corpus). Monotone early decline is not oscillation cure and not quality. Wait for FINAL step=200 then one-shot quality probe.
