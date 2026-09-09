# Optional exact greedy output-head candidate (not deployed)

New files only:
- greedy_pipeline_grouped_model.hpp
- benchmark_greedy_pipeline_grouped.cpp

The owning GreedyPipelineGroupedModel preserves the complete PipelineGroupedModel recurrent implementation and existing step/advance APIs. greedy_step(token,state) returns the next token and advances the same state. It computes final normalization, evaluates every vocabulary row using the same PipelineRows::dot and separate FP32 bias addition as step, checks finiteness for every row including roles, excludes 256/257/258, and resolves equal maxima by lowest token ID. No early termination or approximate pruning, no new quantization, no logits allocation in production greedy execution. Existing compact checkpoint q/scale representation is consumed unchanged.

The optional diagnostic greedy_head_observe(hidden, observer) exposes each scalar for tests without requiring a production logits buffer. Observer callbacks must not throw, recurse into the model, or write shared non-row-local storage. Like existing step, recurrent state has already advanced if head validation rejects a nonfinite scalar. No rollback guarantee is added.

One caller and one persistent worker participate per model; all projections and head share its single executor. Partial reductions write separate cache-line-aligned destinations; dispatch joins before inspecting them. The executor remains the last member, shutting down before owned weights are destroyed. No concurrent destruction or mutation of public weights/config/fast is supported.

Compile the benchmark with C++17, optimization, AVX2, and precise floating-point rules. MSVC example flags: /std:c++17 /O2 /arch:AVX2 /fp:precise. Do not use fast-math/FMA contraction/LTCG to make arithmetic claims. There is no runtime ISA guard: run only on an AVX2-capable CPU/OS, as with the existing pipeline candidate. Parent runs compilation using node child_process. No GPU commands or tests are involved.

Run --unit for synthetic scalar-bit, state/token, finite, role and tie tests. Width 1031 with vocab 263 exceeds the executor threshold; smaller dimensions cover SIMD and scalar tails. Both fast=true and false are tested. Nonfinite cases verify that all row observers fired exactly once before rejection.

Run without arguments from D:/TaoVm for fixed build/yaoyao_graph_step_360.dsb with tokenizer identity 34463f0ac2baf1eb5df792c309a58ea458b109e1187225a0210ccbe89e883333. The printed hash is the required tokenizer identity, not a newly computed checkpoint SHA256. The bundle reader verifies manifest identity and bundle checksum. Two alternating 128-position trials compare stage full-logits plus resident scan, pipeline full-logits plus resident scan, and pipeline greedy. Every trial starts at token 256 with zero state; these are forced generation positions, not end-aware dialogue measurements. CPU timing can be affected by concurrent GPU training, which this benchmark never inspects or touches.

Acceptance claims must be based on actual parent test output. No compilation, runtime pass, or measured speedup was established by the source author. The candidate does not return a logits vector, so do not claim general logits-vector equivalence for its omitted output. Tests compare each computed scalar on fixtures and recurrent states/tokens on the checkpoint; the stage/pipeline full-vector comparison is a separate reference check.
