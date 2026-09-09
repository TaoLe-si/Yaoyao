# GPU fixed validation arch2: source staged, NOT built or accepted

## Latest required deployment policy (supersedes earlier acceptance scheduling below)

Retain ONE Evaluator with preloaded corpus for the trainer lifetime. Immediately after each successful optimizer update, use Result::due(tr.steps): every 10 positive optimizer steps. Each run resets its private validation states. Abandon routine CPU validation completely; no CPU fallback on GPU failure (fail closed and retain last complete checkpoint).

Emit validation.metrics_json() as the controller metric source; exact keys: validation_step (integer), docs (23), supervised (4906), NLL (number), datasetSHA (pinned SHA string), backend, lr_decision_due (boolean). Do not substitute train_preupdate_NLL.

IMPORTANT: accept/record every-10 GPU metrics, but invoke existing stale-count / plateau / LR-decision logic ONLY when validation_step % 100 == 0 (Result::lr_decision_due). Off-cycle observations must not update the LR scheduler best/stale counters either; otherwise decisions still change despite gating LR writes. Preserve prior per-100 decision semantics and separate display/global-best tracking from scheduler state.

Predeployment parity may use EXISTING CPU checkpoint evidence instead of rerunning CPU inference, provided exact checkpoint/step, effective weight representation (no master or changed repacking), arch2 operator, dataset/tokenizer hashes, 23/4906 counts, and metric definition all match. Compare GPU NLL to that evidence within proposed abs tolerance 1e-4; account for printed precision. If provenance is insufficient, do not declare acceptance or silently start routine CPU fallback. The fuller diagnostic procedure below is optional investigation, not a requirement to run new CPU validation.

Parent-only integration after a saved safe boundary:

~~~cpp
// Once, after restore and preflight/stop exits:
using namespace tao::dual::fixed_validation;
Evaluator validation(tr, Corpus("build/bpe_pilot_validation.bin",
    "build/formal_tokenizer.bbp", Corpus::fixed_dataset_sha256, th));
// Immediately after existing tr.update(n,lr), before next training replay:
if (Result::due(tr.steps)) {
    if (slots.active != -1) throw std::runtime_error("validation slot boundary");
    auto v = validation.run();
    auto json = v.metrics_json();
    printf("FIXED_VALIDATION %s\n", json.c_str());
    fflush(stdout);
    // Parent controller consumes all metrics, but changes scheduler best/stale/LR
    // only when v.lr_decision_due(v.step) is true. No routine CPU subprocess.
}
~~~

New implementation: gpu_fixed_validation_arch2.cuh. No live trainer/controller was edited; no GPU test, training, build, or migration was run. Shell pwd fails in this child; absolute reads/writes work. glob discovery fails because ripgrep cannot launch. Official source evaluate_yaoyao_cpu.cpp has now been read completely: arch2 is built by defining TAO_INPUT_SCALE externally. Its fresh-document states, shifted targets, full supervision including TURN_END=259, and stable double CE match this implementation mathematically. Reduction order still requires measured parity. Read CPU model, trainer, reusable graph, batch graph, resident/probe CUDA and batch kernels; legacy CudaResident lacks arch2 input scaling and must not be used as the oracle.

## Backend contract

- Constructor accepts SortedGpuTrainer&, not CPU/master weights. It borrows const pointers to tr.graph.w[name]->value, exactly the projected effective tensors used by ReusableBatchGraph. SortedGpuTrainer::update completes Adam, calls project into these same pointers, then zero_grad. Validator does not call any of those methods.
- No GradNode allocations, gradients, tapes, optimizer, projection, parameter upload/download or CPU forward computation. Existing headers define training kernels but validator launches only forward operators.
- One document per slot, 23 slots. Starts from independent all-zero s/m on every run, teacher-forces every input d[t] to predict d[t+1]; supervision from d[t+1].loss. Never resets inside a document, never truncates a document. Inactive padding cannot update state or counters.
- Arch2 x0=sqrt(D)*effective embedding; tied output embedding is unscaled. RMS epsilon 1e-5; same batch RMS/matvec as training. Memory branches use updated short state and old memory state. Biases/norm gains are borrowed alongside ternary effective matrices.
- Corpus loads existing BPE reader and tokenizer; caller MUST pin independently audited tokenizer and dataset SHA256. Rejects identity differences even for same-size data, and rejects counts other than 23/4906. Raw dataset hash and tokenizer hash returned with result.
- One-time metadata upload only. Each run has zero H2D, no allocations, no parameter transfers, no logits/state readback, one sizeof(Totals) D2H (32 bytes on expected ABI). Per-document CE/count accumulation and final reduction stay on device. FP64 stable log-sum-exp; masked positions contribute zero.
- Counters in Result are API-accounting declarations, not measured profiler counters. Verify using Nsight/CUPTI after training stops. backend=cuda-forward-arch2-effective-resident-v1; actual positions/supervised/bad returned by GPU. NLL=loss/4906.
- No graph capture in this first implementation. Scratch reused per token, O(max dimension * 23), no retained temporal activations. Padding still incurs forward compute. Performance unmeasured.

## Proposed integration only (parent applies after a saved boundary)

Preserve current trainer source and executable. Create a NEW trainer source from complete file contents (never clipped line output), and add this include after existing includes:

~~~cpp
#include "gpu_fixed_validation_arch2.cuh"
~~~

Keep TAO_INPUT_SCALE defined before any model include, existing trainer optimization macros, C++17, and --default-stream per-thread. Link bcrypt.lib as existing tokenizer consumers do. After checkpoint restore, preflight/stop returns, and ReusableBatchGraph construction, construct once:

~~~cpp
using namespace tao::dual::fixed_validation;
Evaluator validation(tr, Corpus("build/bpe_pilot_validation.bin",
    "build/formal_tokenizer.bbp", Corpus::fixed_dataset_sha256, th));
~~~

Corpus::fixed_dataset_sha256 is hard-pinned in the loader to 69900a8acbb28b09296b13a08c76657aa6261c129e929d59bc83ad9d2f1447b0, supplied from parent-audited fixed validation provenance. th is the tokenizer hash already bound into restored trainer checkpoint identity; TLP2 header also binds it. Corpus.tokens and positions derive from parsed documents, not presumed 6413/6390 counts; Result.tokens exposes actual corpus tokens and GPU positions must equal the parsed positions. Do not reconstruct evaluator for each validation. Do not use checkpoint export or CPU bundle in this hotpath.

At the requested cadence immediately AFTER tr.update(n,lr) returns and outside replay/capture (slots.active must be -1), same training host thread, no concurrent streams updating tr:

~~~cpp
if (Result::due(tr.steps)) {
    if (slots.active != -1) throw std::runtime_error("validation slot boundary");
    auto v = validation.run();
    printf("FIXED_VALIDATION step=%u backend=%s docs=%zu supervised=%llu positions=%llu loss_sum=%.17g NLL=%.12f bad=%llu dataset_sha256=%s tokenizer_sha256=%s parameter_h2d_bytes=%zu hotpath_h2d_bytes=%zu d2h_bytes=%zu\n",
        v.step, v.backend, v.documents, v.totals.supervised, v.totals.positions,
        v.totals.loss, v.nll(), v.totals.bad, v.dataset_sha256.c_str(),
        v.tokenizer_sha256.c_str(), v.parameter_h2d_bytes, v.hotpath_h2d_bytes, v.d2h_bytes);
    fflush(stdout);
}
~~~

Validator must die before trainer. Stop/restart evaluator if effective pointers are replaced; pointer checks fail closed. run() does not itself prove external SequenceSlots lifecycle or prevent concurrent writer threads. Caller owns that invariant. Never call on a different host thread with per-thread default stream.

## Optional full diagnostic acceptance procedure (inactive training only)

1. Official evaluate_yaoyao_cpu.cpp has been read and its stable double CE formula verified: max(logits)+log(sum(exp(double(logit)-max)))-logit[target]. Dataset path is build/bpe_pilot_validation.bin; tokenizer is build/formal_tokenizer.bbp; arch2 executable requires externally defined TAO_INPUT_SCALE. No clipping or special-token exclusion. Record actual parsed token/position counts; do not hardcode presumed counts.
2. Restore one immutable saved checkpoint into SortedGpuTrainer using existing identity-aware loader; no update. Construct validator only after restore/project is complete. Record checkpoint hash, step, full Config, dataset SHA, tokenizer SHA. Set TAO_INPUT_SCALE before all CPU and CUDA model includes.
3. TEST-ONLY download tr.graph.w[name]->value once to CpuModel.w, never tr.master. Do not serialize/requantize through a ternary bundle for this test. Compare every copied float bit against that device effective snapshot. This is test instrumentation, not production hotpath.
4. CPU forward oracle: for each document in original order, state=cpu.initial(); for i=1..doc.size()-1, logits=cpu.step(doc[i-1].id,state); increment positions; if doc[i].loss, accumulate stable double CE above and supervised++. Total across all 23 documents. Use scalar CpuModel (no TAO_CPU_AVX2) first, then current official CPU evaluator's implementation on identical effective weights. This is CPU inference ONLY, never CPU training.
5. GPU: a=validation.run(); b=validation.run() without any update. Both documents=23, supervised=4906, positions=CPU positions=sum(doc.size()-1), bad=0, hashes exact. abs(a.nll()-cpu_nll)<=1e-4 and abs(a.loss-cpu_loss)<=0.4906; abs(a.nll()-b.nll())<=1e-12. These are proposed acceptance tolerances, NOT measured parity claims. Fail closed above threshold; never relabel a differing metric as validated. No expected numerical NLL can be asserted before choosing the exact checkpoint.
6. TEST-ONLY bytewise before/after snapshots: all effective tensors, master, moments, variances, scales, all param grads, trainer.graph.s/m, SequenceSlots.s/m, cursor positions and tr.steps unchanged after both runs. CPU oracle gets separate states. Include nonzero seeded training states and gradient buffers in test to detect accidental resets/writes; do not update/optimize those fixtures.
7. Test same-count mutated corpus fails SHA check; wrong tokenizer fails; wrong 4906 count fails; repeated run resets independently; padded shorter docs do not increment counters. On a separate diagnostic run inspect first/last supervised logits/layer states if parity fails; NEVER add full readback to production validator.
8. Profile warm run: no H2D, exactly one 32-byte D2H, no cudaMalloc/free in run, no Adam/projection/backward kernels. Constructor allocation/upload excluded. Record wall time separately from training throughput/checkpoint I/O; do not claim speedup before measurement.
9. Only after all gates pass, migrate NEW trainer executable at persisted optimizer boundary with unchanged checkpoint/control identity/cursor/state and explicit backend logging. Parent owns build commands, executable naming, controller switch, and resumption. Keep official CPU result alongside GPU parity receipt until approved.
