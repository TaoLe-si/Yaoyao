# Independent AVX512 + two-thread advance candidate (source only)

Not compiled or executed in this delegated session. No correctness or performance result claimed. Existing CPU files, all GPU sources, training, and deployment remain untouched.

New sources: advance_row_parallel_avx512_cpu_model.hpp, cpu_ternary_avx512_fixed.hpp, cpu_ternary_avx512_fixed.cpp, benchmark_advance_row_avx512.cpp.

The fixed kernel copies the exact dot expression order from cpu_ternary_avx512.hpp but removes its global mutable selector. AVX512 products feed the original eight-lane accumulator low half then high half. Scale multiplication, scalar tail, reduction, recurrent operations, and FP32 activations are preserved. Candidate reuses CpuRowParallelExecutor unchanged (caller plus one persistent worker, threshold 262144 elements). It is a separate class, not a derived class with accidentally statically bound AVX2 linear calls.

## Parent build via Node exec in an MSVC x64 developer environment

Build from D:/TaoVm. Separate translation units are mandatory to keep AVX512 out of the guarded benchmark and baseline. No /GL or LTCG; no fast math. Suggested commands (parent executes, not executed here):

    cl /nologo /std:c++17 /O2 /EHsc /fp:strict /arch:AVX512 /c cpu_ternary_avx512_fixed.cpp /Fobuild/cpu_ternary_avx512_fixed.obj
    cl /nologo /std:c++17 /O2 /EHsc /fp:strict /arch:AVX2 benchmark_advance_row_avx512.cpp build/cpu_ternary_avx512_fixed.obj /Fobuild/benchmark_advance_row_avx512.obj /Febuild/benchmark_advance_row_avx512.exe
    build/benchmark_advance_row_avx512.exe

The launcher checks CPUID XSAVE/OSXSAVE/AVX, AVX2 and AVX512 F/BW/DQ, and XCR0 mask 0xe6 before loading either model or invoking candidate work. Unsupported exits 77. Since launcher is AVX2-compiled, it is an AVX2-host guard against missing AVX512, not a portable pre-AVX launcher.

Defaults: build/yaoyao_graph_step_360.dsb and tokenizer identity 34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333; optional positional bundle/hash. Compact loader validates identity and payload checksum.

Gate: all finite logits and all s/m states bitwise equal for 32 generated steps against AdvanceRowParallelByteCpuModel (two-thread compact AVX2); independently checks candidate advance states at all 32 positions. Only then exactly two alternating pairs AB/BA, each 128 forced generated positions from fresh state/BOS 256, without EOS early exit. Both models const, no mutable SIMD selector. Records and checks generated traces. Timing excludes loading/state allocation; includes greedy argmax and identical trace recording. No GPU calls, training tests, training control files, extra timing trials, or deployment.

Candidate calls require prior hardware guard. Model owns weights and executor; do not mutate weights or destroy concurrently with calls. Timings on a shared CPU/GPU-training host describe that concurrent environment only. Parent must collect real build/run results before judging acceptance.
