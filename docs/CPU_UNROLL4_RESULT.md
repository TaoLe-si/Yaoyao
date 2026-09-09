# Exact unroll4 candidate

StrictAVX2 /fp:strict build passes tiny-tail checks,32full logits/recurrent states and advance states bitwise; paired128generatedtraces equal. Two AB/BA baseline unroll2 trials418.483/406.138tps vsunroll4 451.857/464.705tps. Aggregate412.218->458.191 (+11.15percent). Frozenstep360, concurrentGPUtraining unchanged. Same single accumulator order, deeper product scheduling only. No wider sweep. Candidate not yet resident-integrated; absolute timing not comparable earlier500tps runs under other hostconditions.
