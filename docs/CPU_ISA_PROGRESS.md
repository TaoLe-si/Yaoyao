# CPU instruction-set investigation

CPUID actual hardware: AVX2, AVX512F/BW/VL/VNNI supported; XCR0=e7 with ZMM state enabled. AMX not selected. No activation quantization or VNNI integer dot introduced.

Exact-order AVX512 byte dot computes16 products per iteration then adds lower8 and upper8 sequentially to AVX2 accumulator, preserving row scaling and original accumulation order.32-step full-model logits/states exact against FP32 AVX2. Alternating forced128-position measurement: AVX2 312.633,295.562; AVX512323.014,319.094 tokens/s. Both paths compiled /arch:AVX512 so this isolates kernel selection, not full AVX2-only binary comparison. GPU training concurrent; modest possible gain, not sufficient to replace broadly portable default.

Files cpu_ternary_avx512.hpp and benchmark_cpu_avx512.cpp remain diagnostic candidates. Two-bit AVX2 implementation being prepared independently. Production resident executable unchanged.
