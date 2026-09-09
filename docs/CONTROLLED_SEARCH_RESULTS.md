# Controlled distance search and key controls

Executed after auxiliary readout training, on the same immutable paired dataset and saved three readouts. Native CUDA training; search/evaluation run native C++ on CPU (compiled with nvcc to share source). Not production training or deployment.

## Search protocol and actual budget
For each of7 query rows, visit15 distances,8 nonidentity signed absolute scale codes per coordinate. Start identity, retain only train-CE improvements. OriginalTCG union<=2 distances and state amplitude sum<=8 enforced, using tds validation. Two orders:ascending versus mt19937 seed7331 shuffled separately for each query. Same840 proposed candidate slots/order/readout. Feasible scored evaluations differ due to path-dependent budget constraints:ascending760,random628. This is same candidate enumeration budget, NOT same number of objective evaluations.

All three readout seeds produce exactly identical selected tables under both orders:
- query900(requested d2):state distance12 code4(+8/32 absolute scale).
- query901(d4):distance3 code1(+1/32).
- query904(d16):distance11 code8(-8/32).
- other queries unchanged.

Seed42 heldout CE changes only d2:3.338750919->3.338156368,d4:3.347053862->3.346948293,d16:3.349950048->3.348362825. Binary accuracy remains around chance, no broad retrieval gain. Other seeds similar. No evidence this order randomization helps this small frozen-linear-readout task. No extrapolation to full production SwiGLU training, other query IDs or exhaustive-pair search. Exact-pair exhaustive search was optional and not run. Query tokens900..906 are synthetic controls; they are not a natural-language learned query instruction.

## Key controls
For heldout paired contexts, no-key replaces answer with same filler3 in A/B; swapped-key inserts alternate answer but keeps original label. All no-key binary accuracies exactly50%, and swapped binary is complement of original due to paired construction.

Auxiliary d32 binary across seeds42/123/2026:original56.64/58.20/58.20%;no-key50/50/50%;swapped43.36/41.80/41.80%. This is a small key-sensitive signal, not robust retrieval:top1 only3.91/5.08/3.52%, below seen-answer chance6.25%. Seeds share identical test contexts and are NOT independent dataset replicates. No significance or architecture scaling claim.

Frozen production head with Wbi gates and identity state gates:full1024 top1=0 on all7 distances; binary original ranges47.66..51.56%. This is an untrained synthetic task with arbitrary query IDs, not a direct natural-language benchmark. Production CE and32-class auxiliary CE use different label spaces and cannot be directly compared as model quality.

## Decision
Do not change live search order or extend horizon based on this run. First correct new-version tokenizer normalization and real story EOS supervision, without changing current token IDs/checkpoints in place. Learned retrieval still unestablished beyond weak paired signal; a linear probe is limited and100 steps do not prove convergence.

## Reproduction and remaining verification
Build:build_controlled_retrieval.bat. Generator takes fresh output directory, trainer/search/controls take that directory. Training and search refuse existing output weights at save; avoid reusing output directory for new configs. Data/weights ignored from Git; manifests retained. Logs local:controlled_gate_search.log,controlled_retrieval_controls.log. Independent numeric and dataset verification executed PASS; see CONTROLLED_NUMERIC_VERIFICATION.md. Parent owns the final verifier.
