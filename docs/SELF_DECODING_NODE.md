# Bounded self-decoding node reference

Implemented verify_self_decoding_node.py as independent mathematical Python reference, not production model. No attention, KV cache, training, or external token sequence access during Cursor.pop().

Node=(packed_recent_tokens,valid_count,trit_state,hash_state). V1024 token uses10 bits; write packed=((packed<<10)|token)&((1<<(10K))-1). Pop token=packed&1023; remaining_payload=packed>>10; trit_prev=mod3(trit-code(token)); hash_prev=inv33*(hash-token) modulo2^32. Cursor valid count decrements and refuses beyond horizon. Live node unchanged.

Tests:96 cases across K1,4,16,64 and empty/short/full/overflow windows, all0/all1023/sequential/random;1360 exact token and prefix-state comparisons. Previously colliding sequences[100,101,67] and[101,67,100] still share trit/hash but DIFFER in node payload and decode correct final tokens. Invalid IDs rejected without mutation. Fixed synthetic D32 ternary codebook, not actual production Q1 in this test.

Proof invariant: payload contains the last min(K,t) tokens in base1024; by mask/shift each pop uniquely recovers the next suffix token. Provided fixed codebook, subtraction/modular inverse restores trit/hash prefix states inductively. AfterK pops, oldest retained prefix state is reconstructed but older tokens are unavailable. Cursor is NOT a fully restored historical node with its old fullK-token payload.

Information is saved explicitly in the node, not magically extracted from the old ambiguous hash. Token payload20 bytes at K16/80 at K64, plus count and state; Python integer/object overhead excluded. This is a10-bit positional encoding reference, NOT a reinterpretation of project macro four-value quantization or replacement of ternary add/sub arithmetic.

Next implementation gates: fixed-width C++ node (or internal packed ring) with cross-word bit tests, production Q1 integration and exact comparisons to forward states, serialization roundtrip, then learning-reader finite differences. This report makes no SIMD/throughput/quality claim. CPU multiplication-free inverse can use a validated fixed shift/add/sub network later; current reference uses integer multiplication by inv33. No production model modified.
