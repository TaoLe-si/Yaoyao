# Tao unified ternary foundation v1

## Scope
Unified discrete value alphabet {-1,0,+1}; NOT all-model quantization completed. New library tao_ternary.hpp provides reusable CPU/CUDA primitives, host validated2bit storage, wide dot and explicit-scale diagnostic quantization. Existing model files, Q1 layout, reader/TCG/TDS, tokenizer IDs, live executable and optimizer untouched. No training or checkpoint conversion performed.

## Encoding
Four elements per byte, lowest2bits first.00=0,01=+1,10=-1,11 invalid. Exact packed length count/4+(count%4!=0). Unused high bits in final byte must be zero. Host validate rejects reserved codes, extra/truncated bytes and noncanonical padding. Empty vectors valid. This new packed layout must NOT be confused with old one-byte q+1 serialization. SelfDecodingNode retains SDN1 layout and token10bit packing, only mod3 arithmetic now delegates to shared primitive. Packed runtime GPU access is unchecked and requires host validation before upload.

## Arithmetic contracts
Chain state:add/subtract mod3 (no scale in ring); valid input trits required. Ordinary dot:wide int64 sum, never mod3. Float activation dot:double accumulation and finite checks, positive finite scale. Gradient/optimizer/logits/softmax/loss remain wider numerical types. Same discrete representation does not imply same accumulation semantics.

QuantizedRow has count,positive finite FP32 row scale,and packed codes. Reference quantizer chooses nearest of{-scale,0,+scale},ties at half scale go to0. Caller explicitly supplies scale; no claim optimal scale estimation or trained quality. Conversion is lossy; not a QAT optimizer and not automatic checkpoint migration. No persistent tensor/checkpoint container introduced yet.

## Tests actually executed
- test_tao_ternary.cpp:known encoding byte0x24,random lengths0..1024 roundtrip and integer dot,illegal values/code/padding/length,INT_MIN/MAX mod3,all9 add/sub inverses,100000-wide accumulation,scale/tie/NaN/Inf rejection:PASS.
- test_tao_ternary_cuda.cu:shared packed decode,ordinary products,mod3 inverse on100003elements CPU/GPU agreement:PASS. Products copied for host wide reduction; no GPU optimized GEMM performance claim.
- test_self_decoding_node.cpp:repaired obsolete archived fresh.bin dependency to current tao_fixed_step50002.bin;K1/4/7/16/32/64 comparisons44/248/560/2144/7360/27008 PASS,packing/serialization/inverse checks PASS.
- test_tao_server_checkpoint:1048576 Wbi equality,100000 sampler draws,4096token vocabulary prefix PASS.
- test_tao_server_forward:617windows631808logits,max_abs3.43322754e-5,top1 identical617/617 PASS (baseline Wbi gates,identity state gates).

## Build
CPU:build_project.bat includes new test_tao_ternary. CUDA:build_ternary_cuda.bat. Binaries in ignored build/.

## Remaining model architecture work
Embedding size/tokenizer,learnable ternary token representation,large-vocabulary Wbi structure,QAT/master weights/scales,activation precision and model container versioning still need architecture decisions. Foundation does not silently quantize current floating heads or replace signed fractional TCG/TDS scales. No new claim of language quality,speed gain,or full ternary deployment.
