# Expanded repair candidate — source only

No build, GPU execution, training or acceptance receipt generation. Only new files were written. Pending parent source review and independent data acceptance.

## CLI

    train_repair_expanded NEW_TRAIN_BIN PARENT_RECEIPT TRUSTED_PARENT_RECEIPT_SHA256 UNIQUE_NEW_CHECKPOINT_DIR UPDATES

Repository-root cwd required; intended input build/repair_export_20260909_v1/train.bin. UPDATES=1..40; output must not exist and its private parent must exist. Same C++17 / --default-stream per-thread CUDA build conventions as native resume test; build/run is a separate parent decision. No resume controller or auto-retry.

## Independent gates

CLI arguments are dataset, receipt, trusted receipt SHA256, unique checkpoint directory, updates. Manifest is automatically read from train.manifest.tsv beside train.bin; no invented completion file or manifest CLI argument. It is parsed as key<TAB>value lines: status must be complete; bin_sha256, tsv_sha256, tokenizer_sha256, validation_sha256 and test_sha256 must match actual artifacts. tokens, supervised and accepted must match parsed corpus token count, target count and document count. Duplicate/missing/malformed fields fail. Additional unique metadata fields are allowed. TSV hashes train.tsv beside train.bin; held-out hashes use build/bpe_pilot_validation.bin and build/bpe_pilot_test.bin.

Parent writes the receipt ONLY after completed export, independent native grammar/full-coverage validation, and composition-tool verification of hashes/counts, plus source review. Exact UTF-8 without BOM, LF endings including final LF, these four lines only:

    TAO_REPAIR_CORPUS_ACCEPT_V1
    dataset_sha256 <64 lowercase hex>
    tokenizer_sha256 <64 lowercase hex>
    manifest_sha256 <64 lowercase hex>

Trainer independently parses the receipt, rejects wrong/missing/reordered/duplicate/extra fields, invalid digests and noncanonical line endings; matches all three hashes against actual corpus bytes, frozen tokenizer identity and exact original manifest bytes. It also checks receipt bytes against the separately parent-supplied trusted digest. Do not compute that trusted argument from an untrusted candidate receipt at launch. Trainer never creates an authorization receipt or offers a bypass. A copy of an already accepted receipt in the final checkpoint is audit evidence only. The simple receipt approves corpus/tokenizer/manifest, not a particular budget/output/source; those remain bounded/fixed by candidate source and parent launch review.

Trainer performs manifest hash/count checks but does not replace independent native grammar/full-coverage and composition validation: trusted parent issuance attests those checks succeeded. Inputs/output parent must remain private and immutable. This is not a signature scheme or hostile-race protection. Partial train.bin.partial/train.tsv.partial must be absent. Existing outputs cannot be reused. STOP_TRAINING must remain. Exact safety checks require build/STOP_TRAINING and reject build/STOP_REPAIR, build/graph_run.lock and build/repair_order.lock. Other lock-named files are not rejected; trainer never removes controls. Parent must externally exclude concurrent production/repair launches.

## Transition and outputs

Load old1200 using OLD canonical corpus and OLD controlled SCP identity before adopting NEW identity. Preserve master weights, Adam m/v, step and native effective projection. Zero slot s/m and discard old scheduler; this is fresh data-domain transition, not old-cursor continuation. Tested exhausted-sentinel initialization of existing ShuffledEpochCursor (alias DeterministicShuffledEpochCursor), seed20260909 epoch1; canonical docs never physically reordered. Rollover only exhausted after prior GPU work synchronizes. Batch plans use the existing tested shuffled_epoch_batch_plan.hpp adapter through tao::data::take_batch, with no local duplicate. Four slots, eight accumulations, width256, fixed LR .00025. Official pinned GPU validation unchanged initially and every update. Initial training NLL explicitly NA with zero consumed targets/positions; initial dataset counts included. Perupdate emits measured preupdate train NLL, targets, positions, validation NLL and norm, plus a FIXED_VALIDATION record using actual Result fields and nll(), matching the production caller API (no metrics_json dependency).

Final native wrapper state.scp / epoch.sidecar / manifest binds NEW dataset SHA, full policy, source SHA and receipt digest via identities. FINAL prints manifest digest; retain it independently for any future separately reviewed native resume. Supplemental pilot.metrics, transition.identity, verified.identity, accepted.receipt and export.manifest are audit files, not covered by existing wrapper manifest. Failed attempts can leave uncommitted or committed-but-audit-incomplete directories; never overwrite/retry. No resume CLI is included. Final quality-probe export final.dsb is also written without overwrite (temporary publication inside new private directory); its SHA256 is printed as QUALITY_EXPORT. This effective-weight DSB is inference-only; native wrapper remains the resumable artifact. It is not covered by the wrapper manifest.

## Review status

Source interfaces inspected; no runtime correctness claims. Shell pwd and glob could not launch in delegated runtime. Earlier export inspection contained train.bin.partial and train.tsv.partial only; subsequent attempt to read actual train.manifest.tsv returned not found. Actual manifest checks implement the parent-specified schema and await completed-export review. No completion or data acceptance claimed.
