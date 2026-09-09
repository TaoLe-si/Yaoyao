# Frozen checkpoint1200 train/validation diagnostic — sources ready, not compiled

New standalone `evaluate_frozen_train_validation_arch2.cu` uses new `gpu_frozen_diagnostic_arch2.cuh`. Production files untouched. Header is a mechanical copy with renamed namespace, caller-selected 23-doc Corpus, dynamic supervised count validation and per-document readback. Forward method is text-identical to production.

Parent build using existing working NVCC environment and GPU architecture flags, C++17, `--default-stream per-thread`, link `bcrypt.lib`. Example command skeleton:

```text
nvcc -std=c++17 --default-stream per-thread evaluate_frozen_train_validation_arch2.cu -o build/evaluate_frozen_train_validation_arch2.exe bcrypt.lib
```

Parent runs ONCE from D:/TaoVm with STOP_TRAINING retained and trainer stopped:

```text
build/evaluate_frozen_train_validation_arch2.exe --diagnostic <checkpoint1200.dsb> <audited-checkpoint-sha256> <audited-train-dataset-sha256> build/frozen1200_train_validation
```

Expected hashes must be lowercase exact SHA256; parent establishes that chosen checkpoint hash really belongs to step1200. DSB manifest binds architecture/tokenizer but does not encode step; 1200 is explicitly a reporting label, not falsely restored optimizer state. Validation and tokenizer identities are pinned to production constants.

Selection sorts the entire training corpus by full token length, ties by original zero-based document ID. Splits sorted ranks into 23 disjoint equal-population strata and selects lower midpoint of each. No model/loss participates. It is a deterministic coverage sample, NOT full training NLL or an unbiased statistical estimate. Official validation uses all 23 docs in original order and requires 4906 targets. Every document is fully teacher-forced with fresh private recurrent state; no truncation, gradients, backward, optimizer update, or checkpoint writes. Each corpus uses reusable fixed23 evaluator scratch.

SortedGpuTrainer constructor necessarily projects and allocates training storage. Immediately afterward ALL graph effective values are overwritten byte-for-byte from load_bundle decoded DSB weights. Bytewise host readback verifies equality before inference and after each corpus. No subsequent project is called; optimizer storage is unused. This retains the required trainer type without allowing constructor projection to change evaluated values.

Before model inference the executable creates NEW `<prefix>.manifest.tsv` with selected IDs, strata boundaries/ranks/min/max lengths, full dataset and checkpoint hashes, tokenizer identity. Refuses overwrite. After success creates `<prefix>.results.tsv`: all 46 per-document rows and aggregate rows with token/position/supervised counts, loss sums and target-weighted NLL. Zero-supervision doc NLL is NA. Stdout gives train_sample NLL, full_validation NLL and gap. Manifest remaining without results means incomplete/failed run.

No GPU execution or build performed by child. Shell pwd fails (File not found); absolute file tools work. Parent must compile/run and inspect DIAGNOSTIC_OK. No measured NLL or compile-success claim yet. STOP file and source checkpoint are only read, never modified.
