# Round66 reader and real-text baseline

Existing files independently validated: train2015docs2951933tokens1798346loss; validation23docs26608tokens19875loss; test20docs24646tokens11500loss. Native reader checks record lengths, truncation, ID/mask ranges, BOS/role/body/turn/EOS structure. Header and truncated-length negative tests passed. No re-export. No UTF8/near-dedup/full fuzz claims.

Train-only assistant-target bigram additive-one smoothing,261 output classes: trainNLL2.694505; validationNLL2.855619 vsuniform5.564520. Bits/supervised-token4.119788 includes special tokens, NOT plain-text bits/byte. Test format checked but test loss not evaluated. Small one-shard pilot, not model ability benchmark. Neural conventional baseline still needed. No generation measured.

Outputs pilot_reader.hpp,pilot_bigram.cpp. Goalactive.
