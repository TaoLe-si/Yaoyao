# Controlled retrieval completion summary

Completed isolated dataset generation, three native CUDA100-step linear-readout probes, independent prefix/logit/gradient checks, production and no-key/swapped-key controls, and ascending/random distance search. See CONTROLLED_SEARCH_RESULTS.md and CONTROLLED_NUMERIC_VERIFICATION.md for actual numbers and limitations. No online model/data changed.

Data:7168 train+1792 disjoint-context test+1792 unseen-answer exploratory records, seven distances and16 balanced answers/split. SHA in experiments/controlled_retrieval_v1/manifest.tsv. No train/test input overlap. Raw datasets/readouts ignored from Git; generator and manifest published.

The proposed shared loader/header refactor was not needed for execution: verifier independently checks the existing shared load() and dataset; all tools include the guarded trainer to reuse data/feature routines. Exact-pair exhaustive search was optional and not run.100 steps are not convergence evidence. Unseen output classes have zero training supervision, so their zero top1 is not a memory conclusion.

Results:near-chance general retrieval, modest d32 paired sensitivity, no difference in selected tables between ascending and randomized order. Negative controls prevent interpreting train loss improvement as successful retrieval. Recommendation:no horizon expansion or live search switch based on this evidence; correct new-version tokenizer normalization and real-story EOS supervision first.

Reproduction:build_controlled_retrieval.bat, make_controlled_retrieval.exe fresh-directory, train_controlled_readout.exe directory, verify_controlled_readout.exe directory, search_controlled_gates.exe directory, controlled_retrieval_controls.exe directory. Training/search now preflight existing outputs before execution. All command executables under build/.
