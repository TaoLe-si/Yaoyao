# Controlled readout numeric verification — executed PASS

Parent replaced an incomplete delegated wrapper before execution. The actual verify_controlled_readout.cu was compiled with MSVC14.44/CUDA12.6,--fmad=false,-O2,sm75 and executed against saved data/readouts; no training weights were overwritten.

- Dataset7168+1792+1792=10752 records:paired metadata, answer substitution position, d64 identical-input control, distance/answer balance, train-vs-test input overlap all PASS.
- Independent forward mod3 saved-prefix arrays and rolling hash vs inverse-pop extracted321 features:maximum absolute error0 across all10752 records. The oracle does not call SelfDecodingNode::pop or push for reconstruction; it uses direct mod3 arithmetic and stored prefix states. Shared Q1 and coefficients remain frozen input assets.
- Posttraining CPU vs GPU logits:all7168 training rows x32classes, all3 saved readouts,maximum error0.
- Independent double-precision CPU softmax gradient/SGD+L2 update vs GPU on64 rows and all10272 parameters:seed42 max7.56655771e-9,seed123 max7.70683248e-9,seed2026 max7.38524483e-9. Tolerance1e-6. Device probe updates temporary copied weights only; saved artifacts unchanged.

Full log:controlled_numeric_verification.log(local ignored). This verifies the limited auxiliary task implementation, not story generation quality or convergence. No independent GPU state kernel was needed because feature extraction is CPU-only and was checked against a saved-prefix oracle.
