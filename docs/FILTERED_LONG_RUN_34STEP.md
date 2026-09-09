# Filtered long-run 34/200 (not complete)

PID 34952 alive, lr=.0001, official 23-doc GPU val every update, no FAIL, production STOP retained.

Initial 6.309478603453 → step34 6.121487898318843. Best step 35 = 6.121487898318843. Delta -0.187991. Last10 mean 6.13388494816769. Rebound steps: 2
- step 29: 6.138190565507 → 6.138343295360 (d=1.527e-4)
- step 32: 6.131257309574 → 6.132906407712 (d=1.649e-3)

Supervised 134310 / 3887672 (~3.45%). Output dir still empty until FINAL.

Tiny rebounds at 29 (+1.5e-4) and 32 (+1.6e-3) recovered; still not oscillation cure and not quality. Wait for FINAL step=200 then one-shot quality probe.
